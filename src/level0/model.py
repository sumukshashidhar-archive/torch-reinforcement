import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearQNet(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the LinearQNet

        Parameters
        ----------
        input_size : int
            The size of the input layer
        hidden_size : int
            The size of the hidden layer
        output_size : int
            The size of the output layer

        Returns
        -------
        None

        Notes
        -----
        There are two linear layers in this network, it is feedforward.
        """
        super().__init__()
        # let us create two linear layers
        self.linear_one = nn.Linear(input_size, hidden_size)
        self.linear_two = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        # let us pass the input through the first linear layer
        x = F.relu(self.linear_one(x))
        # and then through the second linear layer
        x = self.linear_two(x)
        # and return the output
        return x

    def save(self, file_name: str) -> None:
        """
        Save the model

        Parameters
        ----------
        file_name : str
            The name of the file to save the model to

        Returns
        -------
        None
        """
        # let us save the model
        torch.save(self.state_dict(), file_name)



class QTrainer:

    def __init__(self, model, lr: float, gamma: float) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.L1Loss()
    
    def train_step(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            action = (action, )

        # predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()