from portable.option.memory import SetDataset
from portable.option.ensemble.custom_attention import *

from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import os

from portable.option.sets.models import FactoredMinigridImageFeatureExtractor


initiation_positive_files = [
    'resources/factored_minigrid_images/doorkey_opendoor_0_initiation_image_positive.npy',
]

initiation_negative_files = [
    'resources/factored_minigrid_images/doorkey_opendoor_0_initiation_image_negative.npy',
]

test_positive_files = [
    'resources/factored_minigrid_images/doorkey_opendoor_1_initiation_image_positive.npy',
    'resources/factored_minigrid_images/doorkey_opendoor_2_initiation_image_positive.npy',
]

test_negative_files = [
    'resources/factored_minigrid_images/doorkey_opendoor_1_initiation_image_negative.npy',
    'resources/factored_minigrid_images/doorkey_opendoor_2_initiation_image_negative.npy',
]

encoder_save_dir = "runs/custom_attention_test/encoder/6/encoder.ckpt"

if __name__ == "__main__":
    log_dir_base = "runs/custom_attention_test/ensemble"
    log_dir = os.path.join(log_dir_base, "0")
    x = 0
    while os.path.exists(log_dir):
        x += 1
        log_dir = os.path.join(log_dir_base, str(x))

    writer = SummaryWriter(log_dir=log_dir)

    dataset = SetDataset(batchsize=16)

    dataset.add_true_files(initiation_positive_files)
    dataset.add_false_files(initiation_negative_files)

    test_dataset = SetDataset(batchsize=16)

    test_dataset.add_true_files(test_positive_files)
    test_dataset.add_false_files(test_negative_files)

    attention_heads = 6

    model = AttentionEnsembleII(num_attention_heads=attention_heads,
                                num_features=8,
                                num_classes=2,
                                input_dim=1000*8)

    encoder = AutoEncoder(6, 8, 1000)
    
    # feature_extractor = FactoredMinigridImageFeatureExtractor()
    encoder.load_state_dict(torch.load(encoder_save_dir))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = [
        torch.optim.Adam(model.attentions[idx].parameters(), lr=1e-3) for idx in range(attention_heads)
    ]

    device = torch.device("cuda")
    model.to(device)
    encoder.to(device)

    for epoch in range(3000):
        dataset.shuffle()
        loss = np.zeros(attention_heads)
        classifier_losses = np.zeros(attention_heads)
        classifier_acc = np.zeros(attention_heads)
        div_losses = np.zeros(attention_heads)
        acc = 0
        counter = 0
        for b_idx in range(dataset.num_batches):
            counter += 1
            x, y = dataset.get_batch()
            # x = feature_extractor(x)
            x = x.to(device)
            with torch.no_grad():
                x = encoder.feature_extractor(x)
                print(x)
            y = y.to(device)
            pred_y = model(x)
            masks = model.get_attention_masks()
            for att_idx in range(attention_heads):
                b_loss = criterion(pred_y[att_idx], y)
                pred_class = torch.argmax(pred_y[att_idx], dim=1).detach()
                classifier_losses[att_idx] += b_loss.item()
                div_loss = 1.2*divergence_loss(masks, att_idx)
                div_losses[att_idx] += div_loss.item()
                regulariser_loss = l1_loss(masks, att_idx)
                b_loss += div_loss
                b_loss += regulariser_loss*(0.1)
                classifier_acc[att_idx] += (torch.sum(pred_class==y).item())/len(y)
                b_loss.backward()
                optimizers[att_idx].step()
                optimizers[att_idx].zero_grad()
                loss[att_idx] += b_loss.item()
        
        attention_images = []
        masks = model.get_attention_masks()
        cat_masks = torch.cat(masks).squeeze().detach().cpu().numpy()
        fig, ax = plt.subplots()
        ax.matshow(cat_masks)
        for (i, j), z in np.ndenumerate(cat_masks):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        
        img_save_path = os.path.join(log_dir, 'masks.png')
        fig.savefig(img_save_path, bbox_inches='tight')
        plt.close(fig)
        
        for idx in range(attention_heads):
            writer.add_scalar('train_overall_loss/{}'.format(idx), loss[idx]/counter, epoch)
            writer.add_scalar('train_classifier_loss/{}'.format(idx), classifier_losses[idx]/counter, epoch)
            writer.add_scalar('train_divergence_loss/{}'.format(idx), div_losses[idx]/counter, epoch)
            writer.add_scalar('train_accuracy/{}'.format(idx), classifier_acc[idx]/counter, epoch)
            print("att {} epoch {} = class loss: {:.2f} div loss: {:.2f} total loss: {:.2f} acc: {:.2f}".format(idx, 
                                                                    epoch, 
                                                                    classifier_losses[idx]/counter,
                                                                    div_losses[idx]/counter,
                                                                    loss[idx]/counter,
                                                                    classifier_acc[idx]/counter))
        
        print("=====================================")
        
        accuracies = np.zeros(attention_heads)
        negative_accuracy = np.zeros(attention_heads)
        positive_accuracy = np.zeros(attention_heads)
        counter = 0
        for b_idx in range(test_dataset.num_batches):
            # print(b_idx)
            with torch.no_grad():
                x, y = test_dataset.get_batch()
                x = x.to(device)
                x = encoder.feature_extractor(x)
                # x = feature_extractor(x)
                y = y.to(device)
                pred_y = model(x)
                counter += 1
                for att_idx in range(attention_heads):
                    pred_class = torch.argmax(pred_y[att_idx], dim=1).detach()
                    accuracies[att_idx] += torch.sum(pred_class==y).item()/len(y)
                    positive_accuracy[att_idx] += torch.sum(
                        pred_class[y==1]==y[y==1]
                    ).item()/len(y[y==1])
                    negative_accuracy[att_idx] += torch.sum(
                        pred_class[y==0]==y[y==0]
                    ).item()/len(y[y==0])
        
        for attn_idx in range(attention_heads):
            writer.add_scalar('test_overall_accuracy/{}'.format(attn_idx), accuracies[attn_idx]/counter, epoch)
            writer.add_scalar('test_positive_accuracy/{}'.format(attn_idx), positive_accuracy[attn_idx]/counter, epoch)
            writer.add_scalar('test_negative_accuracy/{}'.format(attn_idx), negative_accuracy[attn_idx]/counter, epoch)
            
            
            # img_save_path = os.path.join(log_dir, 'positive_attention_{}.png'.format(attn_idx))
            # plot_attentioned_state(x[y==1][0].cpu(), masks[attn_idx].detach().squeeze().cpu(), img_save_path)
            
            # img_save_path = os.path.join(log_dir, 'negative_attention_{}.png'.format(attn_idx))
            # plot_attentioned_state(x[y==0][0].cpu(), masks[attn_idx].detach().squeeze().cpu(), img_save_path)

