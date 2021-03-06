import torch
import torch.nn as nn
import torch.nn.functional as F

class GE2ELoss(nn.Module):
    def __init__(self, features, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        '''
        The features are of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        '''
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method
        self.features = features # (N, M, D)
        self.speakers = self.features.shape[0]
        self.utterances = self.features.shape[1]
        self.emb_size = self.features.shape[2] 
        self.centroids = torch.mean(self.features, axis=1) # (N, D)
        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_cosine_sim(self):
        #Make the cosine similarity matrix with dims (N,M,N)
        cosine_matrix = torch.zeros(self.speakers, self.utterances, self.speakers)
        for speaker_id, speaker in enumerate(self.features):
            for utterance_id, utterance in enumerate(speaker):
                original_centroid = torch.copy(self.centroids[speaker_id])
                self.centroids[speaker_id] = (torch.sum(self.features[speaker_id], axis=0) - utterance) / (self.utterances - 1)
                cosine_matrix[speaker_id][utterance_id] = torch.clamp(torch.mm(self.centroids, utterance) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6)
                #The own speaker id centroid needs recomputation
                self.centroids[speaker_id] = original_centroid
        return cosine_matrix
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)
        '''


    def embed_loss_softmax(self, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
       # N, M, _ = dvecs.shape
        L = []
        for j in range(self.speakers):
            L_row = []
            for i in range(self.utterances):
                L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, cos_sim_matrix):
        ''' 
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        '''
        #N, M, _ = dvecs.shape
        L = []
        for j in range(self.speakers):
            L_row = []
            for i in range(self.utterances):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        '''
        #Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim()
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(cos_sim_matrix)
        return L.sum()