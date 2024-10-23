import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from MViT_pytorch_upload import MViT
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt

rngsd = 1

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--Dataset', choices=['Houston, Berlin, Augsburg'], default='Augsburg', help='dataset to use')
parser.add_argument('--Flag_test', choices=['test', 'test'], default='train', help='testing mark')
parser.add_argument('--Mode', choices=['MViT'], default='MViT', help='mode choice')
parser.add_argument('--Gpu_id', default='0', help='gpu id')
parser.add_argument('--Seed', type=int, default=1, help='number of seed')
parser.add_argument('--Batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--Test_freq', type=int, default=10, help='number of evaluation')
parser.add_argument('--Patches', type=int, default=13, help='number of patches')
parser.add_argument('--Epoches', type=int, default=500, help='epoch number')
parser.add_argument('--Learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--Gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--Weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

# Create lists to store metrics for plotting
train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []
test_average_accuracies = []

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Initialize the plots
        plt.style.use('seaborn')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training and Testing Metrics')
        
        # Configure subplots
        self.train_acc_line, = self.axes[0, 0].plot([], [], 'b-', label='Train Accuracy')
        self.train_loss_line, = self.axes[0, 1].plot([], [], 'r-', label='Train Loss')
        self.test_oa_line, = self.axes[1, 0].plot([], [], 'g-', label='Test OA')
        self.test_aa_line, = self.axes[1, 0].plot([], [], 'y-', label='Test AA')
        self.test_loss_line, = self.axes[1, 1].plot([], [], 'm-', label='Test Loss')
        
        # Set labels and legends
        self.axes[0, 0].set_title('Training Accuracy')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Accuracy (%)')
        self.axes[0, 0].legend()
        
        self.axes[0, 1].set_title('Training Loss')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].legend()
        
        self.axes[1, 0].set_title('Test Accuracies')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Accuracy (%)')
        self.axes[1, 0].legend()
        
        self.axes[1, 1].set_title('Test Loss')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Loss')
        self.axes[1, 1].legend()
        
        plt.tight_layout()
    
    def update_plots(self, epoch, train_acc, train_loss, test_acc=None, test_aa=None, test_loss=None):
        # Update training metrics
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        epochs = list(range(len(train_accuracies)))
        
        self.train_acc_line.set_data(epochs, train_accuracies)
        self.train_loss_line.set_data(epochs, train_losses)
        
        # Update testing metrics if available
        if test_acc is not None:
            test_accuracies.append(test_acc)
            test_average_accuracies.append(test_aa)
            test_losses.append(test_loss)
            test_epochs = list(range(0, epoch + 1, args.Test_freq))
            
            self.test_oa_line.set_data(test_epochs, test_accuracies)
            self.test_aa_line.set_data(test_epochs, test_average_accuracies)
            self.test_loss_line.set_data(test_epochs, test_losses)
        
        # Update axis limits
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, f'training_metrics_epoch_{epoch}.png'))
        plt.close()

# The rest of your code remains the same until the training loop

if args.Flag_test == 'train':
    # Initialize visualizer
    visualizer = Visualizer(folder_log)
    
    best_checkpoint = {"OA_TE": 0.50}
    
    print("=================================== Training ===================================")
    tic = time.time()
    for epoch in range(args.Epoches): 
        model.train()
        
        train_acc, train_obj, tar_t, pre_t = train_epoch_MM(model, Label_TR_loader, criterion, optimizer)
        scheduler.step()
        OA_TR, AA_TR, Kappa_TR, CA_TR = output_metric(tar_t, pre_t)
        
        # Update plots with training metrics
        visualizer.update_plots(epoch, 
                              train_acc.data.cpu().numpy(),
                              train_obj.data.cpu().numpy())
        
        if (epoch % args.Test_freq == 0) | (epoch == args.Epoches - 1):
            print("Epoch: {:03d} train_loss: {:.4f}, train_OA: {:.2f}".format(epoch+1, train_obj, OA_TR*100))
            
            model.eval()
            test_acc, test_obj, tar_v, pre_v = valid_epoch_MM(model, Label_TE_loader, criterion, optimizer)
            OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(
                epoch+1, train_obj, OA_TE*100, AA_TE*100, Kappa_TE))
            
            # Update plots with test metrics
            visualizer.update_plots(epoch,
                                  train_acc.data.cpu().numpy(),
                                  train_obj.data.cpu().numpy(),
                                  OA_TE*100,
                                  AA_TE*100,
                                  test_obj.data.cpu().numpy())
            
            if OA_TE*100 > best_checkpoint['OA_TE']:
                best_checkpoint = {
                    'epoch': epoch,
                    'OA_TE': OA_TE*100,
                    'AA_TE': AA_TE*100,
                    'Kappa_TE': Kappa_TE,
                    'CA_TE': CA_TE*100
                }
            
            PATH = folder_log + args.Dataset + str(epoch) + '.pt'
            torch.save(model.state_dict(), PATH)

    # Rest of your code remains the same...
            model.eval()
            test_acc, test_obj, tar_v, pre_v = valid_epoch_MM(model, Label_TE_loader, criterion, optimizer)
            OA_TE, AA_TE, Kappa_TE, CA_TE = output_metric(tar_v, pre_v)
            print("Epoch: {:03d} test_loss: {:.4f}, test_OA: {:.2f}, test_AA: {:.2f}, test_Kappa: {:.4f}".format(epoch+1, train_obj, OA_TE*100, AA_TE*100, Kappa_TE))

            vis.line(X=np.array([epoch]), Y=np.array([test_acc.data.cpu().numpy()]), win='test_oa', update='append', opts={'title':'Test Overall Accuracy'})
            vis.line(X=np.array([epoch]), Y=np.array([AA_TE*100]), win='test_aa', update='append', opts={'title':'Test Average Accuracy'})
            vis.line(X=np.array([epoch]), Y=np.array([test_obj.data.cpu().numpy()]), win='test_obj', update='append', opts={'title':'Test Loss'})
            
            if OA_TE*100>best_checkpoint['OA_TE']:
                best_checkpoint = {'epoch': epoch, 'OA_TE': OA_TE*100, 'AA_TE': AA_TE*100, 'Kappa_TE': Kappa_TE, 'CA_TE': CA_TE*100}
                
            PATH = folder_log + args.Dataset + str(epoch) + '.pt'
            torch.save(model.state_dict(), PATH)

    toc = time.time()
    runtime = toc - tic
    print(">>> Training finished!")

    print(">>> Running time: {:.2f}".format(runtime))
    print("=================================== Results ===================================")

    print(">>> The peak performance in terms of OA is achieved at epoch", best_checkpoint['epoch'])
    print("OA: {:.2f} | AA: {:.2f} | Kappa: {:.4f}".format(best_checkpoint['OA_TE'], best_checkpoint['AA_TE'], best_checkpoint['Kappa_TE']))
    np.set_printoptions(precision=2, suppress=True)
    print("CA: ", best_checkpoint['CA_TE'])
    
    output_txt_path = os.path.join(folder_log, 'precision.txt')
    write_message = "Patch size {}, weight decay {}, learning rate {}, the best epoch {}, OA {}, AA {}, Kappa {}, run time {}".format(args.Patches, args.Weight_decay, args.Learning_rate, best_checkpoint['epoch'], round(best_checkpoint['OA_TE'],2), round(best_checkpoint['AA_TE'],2), round(best_checkpoint['Kappa_TE'],4), round(runtime,2))
    
    output_txt_file = open(output_txt_path, "a")
    now = time.strftime("%c")
    output_txt_file.write('=================================== Precision Log (%s) ===================================\n' % now)
    output_txt_file.write('%s\n' % write_message)
    output_txt_file.close()
    
    vis.close('train_acc')
    vis.close('train_obj')
    vis.close('test_acc')
    vis.close('test_obj')
