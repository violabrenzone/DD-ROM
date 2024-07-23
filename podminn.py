from dlroms.roms import ROM
import torch

tanh = torch.nn.Tanh()

class PODMINN(ROM):
    def forward(self, mu, d):
        phi, psi = self[0], self[1]
        out = torch.matmul(psi(d), phi(mu)).squeeze()
        return out if self.trainable else out.mm(self.V.T)
    
    def freeze(self):
        super(PODMINN, self).freeze()
        self.trainable = False