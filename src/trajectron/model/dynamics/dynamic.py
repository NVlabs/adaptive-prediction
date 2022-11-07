class Dynamic(object):
    def __init__(
        self, dyn_limits, device, model_registrar, xz_size, node_type, hyperparams
    ):
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.hyperparams = hyperparams
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x, dt):
        raise NotImplementedError

    def integrate_distribution(self, dist, x, dt):
        raise NotImplementedError

    def create_graph(self, xz_size):
        pass
