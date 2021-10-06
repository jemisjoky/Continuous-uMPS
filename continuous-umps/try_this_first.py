import torch

# After running `make install` in the torchmps folder, this should work
from torchmps import ProbMPS

# Dummy parameters for the model and data
bond_dim = 13
input_dim = 2
batch_size = 55
sequence_len = 21
complex_params = True

# Verify that you can initialize the model
my_mps = ProbMPS(sequence_len, input_dim, bond_dim, complex_params)

# Verify that a Pytorch optimizer initializes properly
optimizer = torch.optim.Adam(my_mps.parameters())

# Create dummy discrete index data (has to be integer/long type!)
data = torch.randint(high=input_dim, size=(sequence_len, batch_size))

# Verify that the model forward function works on dummy data
log_probs = my_mps(data)
assert log_probs.shape == (batch_size,)

# Verify that backprop works fine, and that gradients are populated
loss = my_mps.loss(data)  # <- Negative log likelihood loss
assert all(p.grad is None for p in my_mps.parameters())
# Normally we have to call optimizer.zero_grad before loss.backward, but
# this is just single training run so it doesn't matter
loss.backward()
optimizer.step()
assert all(p.grad is not None for p in my_mps.parameters())

# Congrats, you're ready to start writing the actual training script!
print("Yay, things seem to be working :)")
