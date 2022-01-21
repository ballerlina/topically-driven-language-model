import pdb


class Config:
    def __init__(self, **kwargs):
        def set_required(name):
            if name not in kwargs:
                raise Exception("No {} provided to config.".format(name))
            setattr(self, name, kwargs[name])

        set_required("model_type")
        set_required("num_tm_words")
        set_required("max_seqlen")
        set_required("pad_idx")
        self.beta = kwargs.pop("beta", 1.0)
        self.use_all_bows = kwargs.pop("use_all_bows", False)
        self.eval_false = kwargs.pop("eval_false", False)
        self.reset_hidden = kwargs.pop("reset_hidden", True)
