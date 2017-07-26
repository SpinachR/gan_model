class Config(object):
    def __init__(self, **kwargs):
        # configuration for building the network
        self.x_dim = kwargs.get("x_dim", 784)
        self.x_height = kwargs.get("x_height", 28)
        self.x_width = kwargs.get("x_width", 28)
        self.x_depth = kwargs.get("x_depth", 1)

        self.d_output_dim = kwargs.get("d_output_dim", 10)

        self.c_dim = kwargs.get("c_dim", 12)
        self.z_dim = kwargs.get("z_dim", 100)

        self.batch_size = kwargs.get("batch_size", 128)
        self.unlabeled_batch = kwargs.get("unlabeled_batch", 100)
        self.labeled_batch = kwargs.get("labeled_batch", 28)

        self.lr = kwargs.get("lr", 0.0002)
        self.lr_d = kwargs.get("lr_d", 0.0002)
        self.lr_g = kwargs.get("lr_g", 0.0002)

        self.beta1 = kwargs.get("beta1", 0.5)  # recommend value in dcgan paper

        self.logdir = kwargs.get("logdir", "./logdir")
        self.sampledir = kwargs.get("sampledir", "./sample_image")

        self.max_steps = kwargs.get("max_steps", 100000)
        self.sample_every_n_steps = kwargs.get("sample_every_n_steps", 100)
        self.summary_every_n_steps = kwargs.get("summary_every_n_steps", 5)
        self.savemodel_every_n_steps = kwargs.get("savemodel_every_n_steps", 100)
        self.save_model_secs = kwargs.get("save_model_secs", 1200)

        self.load_ckpt = kwargs.get("load_ckpt", False)
        self.model_ckpt = kwargs.get("model_ckpt", "gcdan.ckpt")
        self.checkpoint_basename = kwargs.get("checkpoint_basename", "./infogan/"+str(self.lr))