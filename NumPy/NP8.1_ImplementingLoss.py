import numpy as np

# Potential values = 0, 1, 2
# Output probabilities for each predicted value for each potential value
softmax_outputs = np.array([[0.7 , 0.1 , 0.2], # {70% --> 0} , 10% --> 1 , 20% --> 2
                            [0.1 , 0.5 , 0.4], # 10% --> 0 , {50% --> 1} , 40% --> 2
                            [0.02 , 0.9 , 0.08]]) # 2% --> 0 , {90% --> 1} , 8% --> 2
class_targets = [0 , 1 , 1] # Resulting highest probability from each of above outputs

print(softmax_outputs[[0 , 1 , 2] , class_targets])
