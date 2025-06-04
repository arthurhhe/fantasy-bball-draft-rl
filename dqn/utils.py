"""Additional utilities for the main code."""

def update_target(model, target_model):
    """Copies the weight from one feedforward network to another.

    Args:
      model (nn.Module): a torch.nn.module instance
      target_model (nn.Module): a torch.nn.module instance
                                from the same parent as model
    """
    target_model.load_state_dict(model.state_dict())
