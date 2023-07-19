import torch 
import numpy as np 
import os
import logging
from portable.option.memory import PositionSet, factored_minigrid_formatter
# from portable.option.sets.models import get_feature_extractor
from portable.option.ensemble.multiheaded_attention import ViT
import copy
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class TransformerSet():
    def __init__(self,
                 device,
                 
                 attention_num,
                 feature_size,
                 feature_extractor_type,
                 encoder_layer_num,
                 feature_num,
                 log_dir,
                 feed_forward_hidden_dim=2048,
                 output_class_num=2,
                 dropout_prob=0.1,
                 feature_extractor_params={},
                 learning_rate=1e-4,
                 key_size=None,
                 
                 dataset_max_size=100000,
                 dataset_batch_size=16):
        
        self.dataset = PositionSet(max_size=dataset_max_size, 
                                   batchsize=dataset_batch_size, 
                                   data_formatter=factored_minigrid_formatter)

        self.feature_num = feature_num
        self.attention_num = attention_num
        self.attention_num = 1
        
        # feature_extractor = get_feature_extractor(feature_extractor_type, feature_extractor_params)
        
        self.vit = ViT(in_channel=6,
                       patch_size=6,
                       feature_dim=feature_size,
                       img_size=84,
                       depth=2,
                       n_classes=2,
                       device=device,
                       **{"attention_num":1,
                          "dropout_prob":0.,
                          "forward_expansion": 4,
                          "forward_dropout":0.})
        
        
        self.device = device
        self.vit.to(device)
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad_clip_norm = 1.0
        self.optimizer = torch.optim.Adam(self.vit.parameters(), learning_rate)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
    def save(self, path):
        classifier_save_path = os.path.join(path, 'classifier')
        torch.save(self.classifier.state_dict(), classifier_save_path)
    
    def load(self, path):
        classifier_save_path = os.path.join(path, 'classifier')
        self.classifier.load_state_dict(torch.load(classifier_save_path))

    def add_data_from_files(self,
                            positive_files,
                            negative_files,
                            priority_negative_files):
        assert isinstance(positive_files, list)
        assert isinstance(negative_files, list)
        assert isinstance(priority_negative_files, list)
        
        self.dataset.add_true_files(positive_files)
        print("false files:", negative_files)
        self.dataset.add_false_files(negative_files)
        print("sanity check")
        self.dataset.add_priority_false_files(priority_negative_files)
        
    def add_data(self,
                 positive_data=[],
                 negative_data=[],
                 priority_negative_data=[]):
        assert isinstance(positive_data, list)
        assert isinstance(negative_data, list)
        assert isinstance(priority_negative_data, list)
        
        if len(positive_data) > 0:
            self.dataset.add_true_data(positive_data)
        
        if len(negative_data) > 0:
            self.dataset.add_false_data(negative_data)
        
        if len(priority_negative_data) > 0:
            self.dataset.add_priority_false_data(priority_negative_data)

    def step(self, x, y=None, epoch=0, plot=False):
        
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        
        if plot:
            output = self.vit((x, y))
        else:
            output = self.vit((x, y))
        
        if y is not None:
            # loss = 0
            # loss_tracker = np.zeros(len(output))
            # acc_tracker = np.zeros(len(output))
            # for idx, attn in enumerate(output):
            #     attn_loss = self.loss(attn, y)
            #     loss += attn_loss
            #     loss_tracker[idx] = attn_loss.item()
            #     pred_class = torch.argmax(attn, dim=1).detach()
            #     acc_tracker[idx] = (torch.sum(pred_class==y).item())/len(attn)
            # """ACCURACY CALC"""
            # loss /= self.attention_num
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=self.grad_clip_norm)
            # self.optimizer.zero_grad()
            # self.optimizer.step()
            # return loss.item(), loss_tracker, acc_tracker

            loss = self.loss(output, y)
            pred_class = torch.argmax(output, dim=1).detach()
            acc = (torch.sum(pred_class==y).item())/len(y)
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            return loss, acc

        else:
            return output


    def train(self, epochs):
        self.vit.train()
        
        for epoch in range(epochs):
            loss = 0
            acc = 0
            loss_tracker = np.zeros(self.attention_num)
            acc_tracker = np.zeros(self.attention_num)
            counter = 0
            self.dataset.shuffle()
            num_batches = self.dataset.num_batches
            for b_idx in range(num_batches):
                x, y = self.dataset.get_batch()
                max_x = torch.max(x)
                if max_x > 1:
                    x /= 255
                
                b_loss, b_acc = self.step(x, y)
                loss += b_loss
                acc += b_acc
                
            loss /= counter +1
            acc /= counter+1
            
            print("loss: {} acc: {}".format(loss, acc))
                
            #     step_loss, attn_losses, attn_accs = self.step(x, 
            #                                                   y, 
            #                                                   epoch,
            #                                                   plot=(b_idx==0))
            #     loss_tracker += attn_losses
            #     acc_tracker += attn_accs
            #     loss += step_loss
            #     counter += 1
            
            # loss /= counter + 1
            # loss_tracker /= counter + 1
            # acc_tracker /= counter + 1
            
            # logger.info("Epoch {}: overall loss: {:.2f}".format(epoch, loss))
            # print("Epoch {}: overall loss: {:.2f}".format(epoch, loss))
            # self.writer.add_scalar("loss/ensemble_loss", loss, epoch)
            # for idx in range(self.attention_num):
            #     logger.info("Att {}: loss={:.2f} accuracy={:.2f}".format(idx, loss_tracker[idx], acc_tracker[idx]))
            #     print("Att {}: loss={:.2f} accuracy={:.2f}".format(idx, loss_tracker[idx], acc_tracker[idx]))
            #     self.writer.add_scalar("loss/member_{}_loss".format(idx), loss_tracker[idx], epoch)
            #     self.writer.add_scalar("accuracy/member_{}_acc".format(idx), acc_tracker[idx], epoch)





