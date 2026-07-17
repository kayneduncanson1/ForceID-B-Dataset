import time
import torch
from Losses import get_dists


# Class from Zhou & Xiang (2019).
# Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch:
# https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/utils/avgmeter.html#AverageMeter
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Class from Rath (2021).
# Using Learning Rate Scheduler and Early Stopping with PyTorch:
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True


def calc_acc_train(dist_ap, dist_an):

    dist_ap_max = torch.max(dist_ap, dim=1)[0]
    indices_dist_ap_max_valid = torch.where(dist_ap_max != float('-inf'))[0]

    if len(indices_dist_ap_max_valid) == 0:

        acc = None
        print('No valid positives in batch')

    else:

        dist_ap_max = dist_ap_max[indices_dist_ap_max_valid].unsqueeze(1)

        # The anchor-negative distance matrix contains negative signed values, so the max operation along the first
        # dimension returns the smallest (i.e., minimum) distance for each anchor. Then, once anchors with valid
        # positives are indexed, the distance values are multiplied by -1 to make their sign positive:
        dist_an_min = torch.max(dist_an, dim=1)[0][indices_dist_ap_max_valid] * -1
        dist_an_min = dist_an_min.unsqueeze(1)

        with torch.no_grad():

            acc = (dist_an_min > dist_ap_max).float().mean().item()

    return acc


# Parameter descriptions available at variable definitions in data_analysis.py:
def train_val(mod, loader_tr, loader_va, labels_va, criterion_tr, criterion_eval, count_samples_min, opt, epochs,
              es_patience, es_min_delta, single_network=True, ptm=True, miner=None, criterion_opt=None, cuda_tr=True,
              cuda_va=True, indices_s1=None, indices_s2=None, n_similarity_search=None):

    # Check the suitability of parameter specifications:
    if not ptm and (miner is not None or criterion_opt is not None):

        raise Exception('Currently only supports miners and loss function optimisers from the ptm package. Check if the'
                        'ptm variable should be set to True in data_analysis.py.')

    elif count_samples_min is not None and indices_s1 is not None:

        raise Exception('count_samples_min and session indices must not be specified simultaneously.')

    elif count_samples_min is None and indices_s1 is None:

        raise Exception('Either count_samples_min or session indices must be specified.')

    elif count_samples_min is None and n_similarity_search is not None:

        raise Exception('count_samples_min and n_similarity_search must be specified simultaneously.')

    # Initialise variables for results to return across training and validation epochs:
    early_stopping = EarlyStopping(patience=es_patience, min_delta=es_min_delta)
    t_start = time.time()
    hist_acc_tr = []
    hist_acc_va = []
    hist_loss_tr = []
    hist_loss_va = []
    acc_best_tr = 0
    acc_best_va = 0

    # Set a baseline checkpoint for model and optimiser states:
    checkpoint = {'model_state_dict': mod.state_dict(), 'optimizer_state_dict': opt.state_dict()}

    for epoch in range(1, epochs + 1):

        for phase in ['train', 'val']:

            if phase == 'train':

                # Initialise a running average meter for the loss and accuracy across training (mini-)batches:
                losses = AverageMeter()
                accs = AverageMeter()

                if cuda_tr:

                    mod.cuda().train()

                else:

                    mod.train()

                # For the single network specification (as was used in the technical validation), there is only one
                # set of inputs for the batch:
                if single_network:

                    for batch_idx, (inputs, labs) in enumerate(loader_tr):

                        if cuda_tr:

                            inputs, labs = inputs.cuda(), labs.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(True):

                            # Get the gait embeddings (i.e., feature sets) for input gait samples in the batch:
                            embs = mod(inputs, cuda_tr)

                            # Get the pair-wise anchor-positive and anchor-negative distance matrices for embeddings in
                            # the batch:
                            dist_ap, dist_an = get_dists(embs, labs, 'train')

                            if ptm and miner is None:

                                # In this case, criterion_tr is a loss function class from the pytorch_metric_learning
                                # (ptm) package:
                                loss = criterion_tr(embs, labs)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif ptm and miner is not None:

                                miner_output = miner(embs, labs)
                                loss = criterion_tr(embs, labs, miner_output)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif not ptm:

                                # If data_analysis.py is modified to use a custom loss function class, it is
                                # expected to be a distance metric loss function that inputs anchor-positive and
                                # anchor-negative distance matrices then returns loss and accuracy:
                                loss, acc = criterion_tr(dist_ap, dist_an)

                            loss.backward()

                            if criterion_opt is not None:

                                criterion_opt.step()

                            opt.step()

                            # Update the running average meters for loss and accuracy:
                            losses.update(loss.detach().cpu().numpy().item())

                            if acc is not None:

                                accs.update(acc)

                    # We don't use a learning rate scheduler, but one could be initialised in data_analysis.py and then
                    # passed into a modified version of this function with a schedular variable. Then, the scheduler
                    # step (scheduler.step()) could be performed here.

                    hist_loss_tr.append(losses.avg)
                    hist_acc_tr.append(accs.avg)

                    if accs.avg > acc_best_tr:

                        acc_best_tr = accs.avg

                # For the Siamese network specification (i.e., single_network = False), there are two sets of inputs for
                # the batch:
                else:

                    for batch_idx, (inputs1, inputs2, labs) in enumerate(loader_tr):

                        if cuda_tr:

                            inputs1, inputs2, labs = inputs1.cuda(), inputs2.cuda(), labs.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(True):

                            # The model takes two sets of inputs and conducts output feature level fusion to return one
                            # set of embeddings:
                            embs = mod(inputs1, inputs2, cuda_tr)

                            # The rest of this loop is as per the single network specification:
                            dist_ap, dist_an = get_dists(embs, labs, 'train')

                            if ptm and miner is None:

                                loss = criterion_tr(embs, labs)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif ptm and miner is not None:

                                miner_output = miner(embs, labs)
                                loss = criterion_tr(embs, labs, miner_output)
                                acc = calc_acc_train(dist_ap, dist_an)

                            elif not ptm:

                                loss, acc = criterion_tr(dist_ap, dist_an)

                            loss.backward()

                            if criterion_opt is not None:

                                criterion_opt.step()

                            opt.step()

                            losses.update(loss.detach().cpu().numpy().item())

                            if acc is not None:

                                accs.update(acc)

                    hist_loss_tr.append(losses.avg)
                    hist_acc_tr.append(accs.avg)

                    if accs.avg > acc_best_tr:

                        acc_best_tr = accs.avg

            else: # phase == 'val'

                if cuda_va:

                    mod.cuda().eval()

                else:

                    mod.cpu().eval()

                # Initialise the validation set embeddings to None and then include embeddings one batch at a time. This
                # avoided our computer exceeding its memory limit by attempting to pass the entire validation set
                # through the model at once:
                embs = None

                if single_network:

                    for batch_idx, inputs in enumerate(loader_va):

                        if cuda_va:

                            inputs = inputs.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(False):

                            outputs = mod(inputs, cuda_va)
                            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                else:

                    for batch_idx, (inputs1, inputs2) in enumerate(loader_va):

                        if cuda_va:

                            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

                        opt.zero_grad()

                        with torch.set_grad_enabled(False):

                            outputs = mod(inputs1, inputs2, cuda_va)
                            embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

                # Set the outputs variable that is no longer needed to None to save memory for calculating pair-wise
                # distance matrices:
                outputs = None

                if cuda_va:

                    dist_ap, dist_an = get_dists(embs, labels_va.cuda(), 'val')

                else:

                    dist_ap, dist_an = get_dists(embs, labels_va, 'val')

                # Perform similarity search to calculate validation loss and accuracy. In the current implementation,
                # the first conditional is satisfied ('if' statement) as count_samples_min=count_va_min from
                # data_analysis.py and n_similarity_search=None:
                if count_samples_min is not None and n_similarity_search is None:

                    loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min)

                elif count_samples_min is not None and n_similarity_search is not None:

                    loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min, n_similarity_search)

                elif count_samples_min is None and indices_s1 is not None:

                    loss, acc = criterion_eval(dist_ap, dist_an, indices_s1, indices_s2)

                # Set pair-wise distance matrices to None before commencing the next epoch to save memory:
                dist_ap = None
                dist_an = None

                hist_loss_va.append(loss.detach().cpu().numpy().item())
                hist_acc_va.append(acc)

                if acc > acc_best_va:

                    acc_best_va = acc

                    # Save the states for the model and optimiser (i.e., parameter settings) that led to the highest
                    # validation accuracy:
                    checkpoint = {'model_state_dict': mod.state_dict(), 'optimizer_state_dict': opt.state_dict()}

        early_stopping(loss)

        if early_stopping.early_stop:

            break

    time_elapsed = time.time() - t_start

    return time_elapsed, hist_acc_tr, hist_acc_va, hist_loss_tr, hist_loss_va, checkpoint['model_state_dict'],\
        checkpoint['optimizer_state_dict']


# Parameter descriptions available at variable definitions in data_analysis.py:
def test(mod, loader_te, labels_te, criterion_eval, count_samples_min, opt, single_network=True, cuda=True,
         indices_s1=None, indices_s2=None, n_similarity_search=None):

    # Check the suitability of parameter specifications:
    if count_samples_min is not None and indices_s1 is not None:

        raise Exception('count_samples_min and session indices must not be specified simultaneously.')

    elif count_samples_min is None and indices_s1 is None:

        raise Exception('Either count_samples_min or session indices must be specified.')

    elif count_samples_min is None and n_similarity_search is not None:

        raise Exception('count_samples_min and n_similarity_search must be specified simultaneously.')

    # Allocate model to cuda or CPU:
    if cuda:

        mod.cuda().eval()

    else:

        mod.cpu().eval()

    # Initialise the test set embeddings to None and then include embeddings one batch at a time. This avoided our
    # computer exceeding its memory limit by attempting to pass the entire test set through the model at once:
    embs = None

    if single_network:

        for batch_idx, inputs in enumerate(loader_te):

            if cuda:

                inputs = inputs.cuda()

            opt.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = mod(inputs, cuda)
                embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    else:

        for batch_idx, (inputs1, inputs2) in enumerate(loader_te):

            if cuda:

                inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

            opt.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = mod(inputs1, inputs2, cuda)
                embs = torch.cat((embs, outputs), dim=0) if embs is not None else outputs

    # Set the outputs variable that is no longer needed to None to save memory for calculating pair-wise distance
    # matrices:
    outputs = None

    if cuda:

        dist_ap, dist_an = get_dists(embs, labels_te.cuda(), 'test')

    else:

        dist_ap, dist_an = get_dists(embs, labels_te, 'test')

    # Perform similarity search to calculate test loss and accuracy. In the current implementation, the first
    # conditional is satisfied ('if' statement) as count_samples_min=count_te_min from data_analysis.py and
    # n_similarity_search=None:
    if count_samples_min is not None and n_similarity_search is None:

        loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min)

    elif count_samples_min is not None and n_similarity_search is not None:

        loss, acc = criterion_eval(dist_ap, dist_an, count_samples_min, n_similarity_search)

    elif count_samples_min is None and indices_s1 is not None:

        loss, acc = criterion_eval(dist_ap, dist_an, indices_s1, indices_s2)

    else:

        # In this case, an Exception should already have been raised based on the parameter specification check at the
        # top of the function. Nevertheless, an Exception is included here to avoid the warning that loss and acc
        # variables could be referenced before assignment:
        raise Exception("Specification across count_samples_min, n_similarity_search, indices_s1 and indices_s2"
                        "parameters is unsuitable.")

    dist_ap = None
    dist_an = None

    return loss.cpu().numpy().item(), acc, embs
