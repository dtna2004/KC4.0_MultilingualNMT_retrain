from modules.optim.adam import AdamOptim
from modules.optim.adabelief import AdaBeliefOptim
from modules.optim.adamw import AdamWOptim
from modules.optim.adabeliefw import AdaBeliefWOptim
from modules.optim.scheduler import ScheduledOptim

optimizers = {"Adam": AdamOptim, "AdaBelief": AdaBeliefOptim, "AdamW": AdamWOptim, "AdaBeliefW": AdaBeliefWOptim}
