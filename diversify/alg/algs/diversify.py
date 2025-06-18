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
            args.featurizer_out_dim, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)

        self.bottleneck = common_network.feat_bottleneck(
            args.featurizer_out_dim, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(args.bottleneck, args.num_classes)

        self.abottleneck = common_network.feat_bottleneck(
            args.featurizer_out_dim, args.bottleneck, args.layer)
        self.aclassifier = common_network.feat_classifier(args.bottleneck, args.num_classes * args.latent_domain_num)

        # Placeholder for dclassifier to be initialized later
        self.dclassifier = None
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)

        self.args = args
        self.dclassifier_initialized = False

    # UPDATED _initialize_dclassifier method to accept a dataloader instead of tensor
    def _initialize_dclassifier(self, loader):
        if not self.dclassifier_initialized:
            self.featurizer.eval()
            self.dbottleneck.eval()
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].cuda().float()  # assuming batch = (inputs, labels, domains, ...)
                    z1 = self.dbottleneck(self.featurizer(x))
                    z1_dim = z1.shape[1]
                    self.dclassifier = common_network.feat_classifier(z1_dim, self.args.latent_domain_num).cuda()
                    self.dclassifier_initialized = True
                    print(f"[INIT] dclassifier initialized with input dim: {z1_dim}")
                    break  # only need one batch to initialize
            self.featurizer.train()
            self.dbottleneck.train()

    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].cuda().float()
        all_c1 = minibatch[1].cuda().long()
        all_d1 = minibatch[2].cuda().long() % self.args.latent_domain_num

        z1 = self.dbottleneck(self.featurizer(all_x1))
        # initialize dclassifier if not done yet
        if not self.dclassifier_initialized:
            self._initialize_dclassifier_from_tensor(z1)  # Remove this or change to always call with loader

        assert z1.shape[1] == self.dclassifier.fc.in_features, f"Shape mismatch: z1={z1.shape[1]} vs fc.in={self.dclassifier.fc.in_features}"

        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1)

        cd1 = self.dclassifier(z1)

        if all_d1.min() < 0 or all_d1.max() >= self.args.latent_domain_num:
            raise ValueError("Domain label out of range for dclassifier!")

        ent_loss = Entropylogits(cd1) * self.args.lam + F.cross_entropy(cd1, all_d1)

        loss = ent_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float()
        all_c = minibatches[1].cuda().long()
        all_d = minibatches[2].cuda().long() % self.args.latent_domain_num
        all_y = all_d * self.args.num_classes + all_c

        print("=== DEBUG: Class Label Check in update_a ===")
        print("all_y.min():", all_y.min().item(), " | all_y.max():", all_y.max().item())
        print("Expected total_classes:", self.args.num_classes * self.args.latent_domain_num)
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
