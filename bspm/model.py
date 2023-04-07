import world
import torch
import time
from dataloader import BasicDataset
from torch import nn
import scipy.sparse as sp
import numpy as np
from sparsesvd import sparsesvd

from torchdiffeq import odeint

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
class LGCN_IDE(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsr()
        end = time.time()
        print('training time for LGCN-IDE', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users,:].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'gowalla'):
            U_2 = U_1 @ norm_adj.T @ norm_adj
            return U_2
        else:
            return U_1
        
class GF_CF(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat
        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, 256)
        end = time.time()
        print('training time for GF-CF', end-start)
        
    def getUsersRating(self, batch_users, ds_name):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if(ds_name == 'amazon-book'):
            ret = U_2
        else:
            U_1 = batch_test @  self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            ret = U_2 + 0.3 * U_1
        return ret
        
        
class BSPM(object):
    def __init__(self, adj_mat, config:dict):
        self.adj_mat = adj_mat
        self.config = config

        self.idl_solver = self.config['solver_idl']
        self.blur_solver = self.config['solver_blr']
        self.sharpen_solver = self.config['solver_shr']
        print(f"IDL: {self.idl_solver}, BLR: {self.blur_solver}, SHR: {self.sharpen_solver}")
        
        self.idl_beta = self.config['idl_beta']
        self.factor_dim = self.config['factor_dim']
        print(r"IDL factor_dim: ",self.factor_dim)
        print(r"IDL $\beta$: ",self.idl_beta)
        idl_T = self.config['T_idl']
        idl_K = self.config['K_idl']
        
        blur_T = self.config['T_b']
        blur_K = self.config['K_b']
        
        sharpen_T = self.config['T_s']
        sharpen_K = self.config['K_s']

        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float()
        print("idl time: ",self.idl_times)
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float()
        print("blur time: ",self.blurring_times)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float()
        print("sharpen time: ",self.sharpening_times)

        self.final_sharpening = self.config['final_sharpening']
        self.sharpening_off = self.config['sharpening_off']
        self.t_point_combination = self.config['t_point_combination']
        print("final_sharpening: ",self.final_sharpening)
        print("sharpening off: ",self.sharpening_off)
        print("t_point_combination: ",self.t_point_combination)

        
    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)
        end = time.time()
        print('training time for BSPM', end-start)
    
    def IDLFunction(self, t, r):
        out = r.numpy() @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
        out = out - r.numpy()
        return torch.Tensor(out)

    def blurFunction(self, t, r):
        R = self.norm_adj
        out = r.numpy() @ R.T @ R
        out = out - r.numpy()
        return torch.Tensor(out)
    
    def sharpenFunction(self, t, r):
        R = self.norm_adj
        out = r.numpy() @ R.T @ R
        return torch.Tensor(-out)

    def getUsersRating(self, batch_users, ds_name):
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users,:].todense())

        with torch.no_grad():
            if(ds_name != 'amazon-book'):
                idl_out = odeint(func=self.IDLFunction, y0=torch.Tensor(batch_test), t=self.idl_times, method=self.idl_solver)
                
            blurred_out = odeint(func=self.blurFunction, y0=torch.Tensor(batch_test), t=self.blurring_times, method=self.blur_solver)
            
            if self.sharpening_off == False:
                if self.final_sharpening == True:
                    if(ds_name != 'amazon-book'):
                        sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out[-1]+blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)
                    elif(ds_name == 'amazon-book'):
                        sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)
                else:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out[-1], t=self.sharpening_times, method=self.sharpen_solver)
        
        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 =  torch.mean(torch.cat([blurred_out[1:,...],sharpened_out[1:,...]],axis=0),axis=0)
            else:
                U_2 =  torch.mean(blurred_out[1:,...],axis=0)
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
            else:
                U_2 = blurred_out[-1]
        
        if(ds_name == 'amazon-book'):
            ret = U_2.numpy()
        else:
            if self.final_sharpening == True:
                if self.sharpening_off == False:
                    ret = U_2.numpy()
                elif self.sharpening_off == True:
                    ret = U_2.numpy() + self.idl_beta * idl_out[-1].numpy()
            else:
                ret = U_2.numpy() + self.idl_beta * idl_out[-1].numpy()
        return ret

class BSPM_TORCH(object):
    def __init__(self, adj_mat, config:dict):
        self.adj_mat = adj_mat
        self.config = config

        self.idl_solver = self.config['solver_idl']
        self.blur_solver = self.config['solver_blr']
        self.sharpen_solver = self.config['solver_shr']
        print(f"IDL: {self.idl_solver}, BLR: {self.blur_solver}, SHR: {self.sharpen_solver}")
        
        self.idl_beta = self.config['idl_beta']
        self.factor_dim = self.config['factor_dim']
        print(r"IDL factor_dim: ",self.factor_dim)
        print(r"IDL $\beta$: ",self.idl_beta)
        idl_T = self.config['T_idl']
        idl_K = self.config['K_idl']
        
        blur_T = self.config['T_b']
        blur_K = self.config['K_b']
        
        sharpen_T = self.config['T_s']
        sharpen_K = self.config['K_s']

        self.device = config['device']
        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float().to(self.device)
        print("idl time: ",self.idl_times)
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float().to(self.device)
        print("blur time: ",self.blurring_times)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float().to(self.device)
        print("sharpen time: ",self.sharpening_times)

        self.final_sharpening = self.config['final_sharpening']
        self.sharpening_off = self.config['sharpening_off']
        self.t_point_combination = self.config['t_point_combination']
        print("final_sharpening: ",self.final_sharpening)
        print("sharpening off: ",self.sharpening_off)
        print("t_point_combination: ",self.t_point_combination)


    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat
        if self.config['dataset'] != 'amazon-book':
            ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)
            del ut
            del s

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)

        if self.config['dataset'] != 'amazon-book':
            left_mat = self.d_mat_i @ self.vt.T
            right_mat = self.vt @ self.d_mat_i_inv
            self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)
        end = time.time()
        print('pre-processing time for BSPM', end-start)
    
    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out

    def getUsersRating(self, batch_users, ds_name):
        batch_test = batch_users.to_sparse()

        with torch.no_grad():
            if(ds_name != 'amazon-book'):
                idl_out = torch.mm(batch_test, self.left_mat @  self.right_mat)

            if(ds_name != 'amazon-book'):
                blurred_out = torch.mm(batch_test, self.linear_Filter)
            else:
                blurred_out = torch.mm(batch_test.to_dense(), self.linear_Filter)

            del batch_test
            
            if self.sharpening_off == False:
                if self.final_sharpening == True:
                    if(ds_name != 'amazon-book'):
                        sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out+blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
                    elif(ds_name == 'amazon-book'):
                        sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
                else:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
        
        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 =  torch.mean(torch.cat([blurred_out.unsqueeze(0),sharpened_out[1:,...]],axis=0),axis=0)
            else:
                U_2 =  blurred_out
                del blurred_out
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
                del sharpened_out
            else:
                U_2 = blurred_out
                del blurred_out
        
        if(ds_name == 'amazon-book'):
            ret = U_2
            del U_2
        else:
            if self.final_sharpening == True:
                if self.sharpening_off == False:
                    ret = U_2
                elif self.sharpening_off == True:
                    ret = self.idl_beta * idl_out + U_2
            else:
                ret = self.idl_beta * idl_out + U_2
        return ret

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))