import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from IPython.display import clear_output
from sklearn.metrics import *

class NetRunner():
    '''Base class to teach and use NN-models'''
    def __init__(self, model, device, opt=None, checkpoint_name=None):
        self.model = model
        self.opt = opt
        self.device = device
        self.checkpoint_name = checkpoint_name
        
        self.epoch = 0
        self.output = None
        self.metrics = None
        self._global_step = 0
        self._set_events()
        self._top_val_accuracy = -1
        self.log_dict = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def _set_events(self):
        '''Initialize dict of scores in the beginning of epoch'''
        self._phase_name = ''
        self.events = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
    
    def _reset_events(self, event_name):
        '''Reset scores dict'''
        self.events[event_name] = defaultdict(list)
        
    def forward(self, img_batch, **kwargs):
        '''Forward pass through model'''
        logits = self.model(img_batch)
        output = {
            'logits': logits
        }
        return output
    
    def run_criterion(self, batch):
        raise NotImplementedError('Metod run_criterion should be implemented')
        
    def output_log(self):
        raise NotImplementedError('Method output_log should be implemented')
        
    def save_chekpoint(self):
        raise NotImplementedError('Method save_chekpoint should be implemented')
    
    def _run_batch(self, batch):
        '''Forward run of 1 batch'''
        X_batch, y_batch = batch
        self._global_step += len(y_batch)
        X_batch = X_batch.to(self.device)
        self.output = self.forward(X_batch)
    
    def _run_epoch(self, loader, train_phase=True, output_log=False, **kwargs):
        '''Run 1 epoch
        loader: batches generator
        train_phase: indicate train or validation phase
        output_log: print log
        '''
        self.model.train(train_phase)
        
        _phase_description = 'Training' if train_phase else 'Evaluation'
        for batch in tqdm(loader, desc=_phase_description):
            if train_phase:
                self.opt.zero_grad()
            self._run_batch(batch)
            with torch.set_grad_enabled(train_phase):
                loss = self.run_criterion(batch)
            if train_phase:
                loss.backward()
                self.opt.step()
        self.log_dict[self._phase_name].append(np.mean(self.events[self._phase_name]['loss']))
        
        if output_log:
            self.output_log(**kwargs)
    
    def train(self, train_loader, val_loader, n_epochs, model=None, opt=None, **kwargs):
        '''Train model
        train_loader, val_loader: batch generators
        n_epochs: number of epochs
        opt: optimizer
        average: average for sklearn multiclass scores
        visualize: True means plot train/val curves
        '''
        self.opt = opt or self.opt
        self.model = model or self.model
        
        for _epoch in range(n_epochs):
            self.epoch += 1
            print("Epoch {} of {} started.".format(self.epoch, n_epochs))
            
            # training part
            self._set_events()
            self._phase_name = 'train'
            self._run_epoch(train_loader, train_phase=True)
            
            # validatiion part
            self._phase_name = 'val'
            self.validate(val_loader, **kwargs)
            self.save_checkpoint()
    
    @torch.no_grad()
    def validate(self, loader, model=None, phase_name='val', **kwargs):
        self.model = model or self.model
        self._phase_name = phase_name
        self._reset_events(phase_name)
        self._run_epoch(loader, train_phase=False, output_log=True, **kwargs)
        return self.metrics

class CNNetRunner(NetRunner):
    def run_criterion(self, batch):
        _, label_batch = batch
        label_batch = label_batch.to(self.device)
        logits = self.output['logits']
        
        loss = nn.CrossEntropyLoss()(logits, label_batch)
        
        scores = F.softmax(logits, 1).detach().cpu().numpy()
        labels = label_batch.numpy().tolist()
        
        self.events[self._phase_name]['loss'].append(loss.detach().cpu().numpy())
        self.events[self._phase_name]['scores'].append(scores)
        self.events[self._phase_name]['labels'].extend(labels)
        
        return loss
    
    def save_checkpoint(self):
        val_accuracy = self.metrics['accuracy']
        if val_accuracy > self._top_val_accuracy and self.checkpoint_name is not None:
            self._top_val_accuracy = val_accuracy
            with open(self.checkpoint_name, 'wb') as file:
                torch.save(self.model, file)
    
    def output_log(self, **kwargs):
        
        scores = np.concatenate(tuple(self.events[self._phase_name]['scores']), axis=0)
        labels = np.array(self.events[self._phase_name]['labels'])
        
        assert len(scores) > 0, print('Scores are empty')
        assert len(labels) > 0, print('Labels are empty')
        assert len(scores)==len(labels), print('Scores and labels have different size')
        
        visualize = kwargs.get('visualize', False)
        if visualize: clear_output()
        
        average = kwargs.get('average', 'binary')
        self.metrics = {
            'loss': np.mean(self.events[self._phase_name]['loss']),
            'accuracy': accuracy_score(labels, scores.argmax(axis=1)),
            'f1_score': f1_score(labels, scores.argmax(axis=1), average=average)
        }
        
        print('{}: '.format(self._phase_name), end='')
        print(' | '.join(['{}: {:.4f}'.format(k, v) for k, v in self.metrics.items()]))
        
        if visualize:
            plt.figure(figsize=(8,5))
            plt.plot(self.log_dict['train'], color='b', label='train')
            plt.plot(self.log_dict['val'], color='c', label='val')
            plt.legend()
            plt.title('Train/val loss.')
            plt.show()       