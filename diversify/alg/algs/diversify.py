# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits

class Diversify(Algorithm):
    def __init__(self, args):
        super(Diversify, self).__init__(args)

        self.featurizer = get_fea(args)

        self.dbottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)

        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
    input_dim=args.bottleneck, num_classes=args.num_classes)

        self.abottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(
    input_dim=args.bottleneck, num_classes=args.num_classes * args.latent_domain_num)

        self.dclassifier = common_network.feat_classifier(
    input_dim=args.bottleneck, num_classes=args.latent_domain_num)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)

        self.args = args

    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].cuda().float()
        all_c1 = minibatch[1].cuda().long()  # class label
        all_d1 = minibatch[2].cuda().long()  # domain/person ID

        # Debug for label range
        if all_c1.min() < 0 or all_c1.max() >= self.args.num_classes:
            print("=== [ERROR] Invalid class label in update_d ===")
            print("all_c1.min():", all_c1.min().item())
            print("all_c1.max():", all_c1.max().item())
            print("Expected num_classes:", self.args.num_classes)
            raise ValueError("Invalid class labels for cross_entropy in update_d()")

        z1 = self.dbottleneck(self.featurizer(all_x1))
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')

        cd1 = self.dclassifier(z1)
        print(f"=== DEBUG: Domain Label Check in update_d ===")
        print(f"all_d.min(): {all_d.min().item()}  | all_d.max(): {all_d.max().item()}")
        print(f"Expected domain_num: {self.args.domain_num}")
        assert all_d.max() < self.args.domain_num, f"Found domain label {all_d.max().item()} >= domain_num"

        ent_loss = Entropylogits(cd1) * self.args.lam + F.cross_entropy(cd1, all_d)

        loss = ent_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.cuda().float()
                index = data[-1]
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')
        print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def update(self, data, opt):
        all_x = data[0].cuda().float()
        all_y = data[1].cuda().long()
        all_d = data[2].cuda().long()

        all_z = self.bottleneck(self.featurizer(all_x))

        disc_input = Adver_network.ReverseLayerF.apply(all_z, self.args.alpha)
        disc_out = self.discriminator(disc_input)

        disc_loss = F.cross_entropy(disc_out, all_d)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)

        loss = classifier_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float()
        all_c = minibatches[1].cuda().long()  # activity label
        all_d = minibatches[4].cuda().long()  # domain label

        # ✅ USE ONLY CLASS LABELS
        all_y = all_c

        print("=== DEBUG: Class Label Check in update_a ===")
        print(f"all_y.min(): {all_y.min().item()}  | all_y.max(): {all_y.max().item()}")
        print(f"Expected total_classes: {self.args.num_classes}")
        assert all_y.max() < self.args.num_classes, f"Found invalid label: {all_y.max().item()}"
        assert all_y.min() >= 0, "Label contains negative index!"

        all_z = self.abottleneck(self.featurizer(all_x))
        all_preds = self.aclassifier(all_z)

        classifier_loss = F.cross_entropy(all_preds, all_y)
        opt.zero_grad()
        classifier_loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}



    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))

