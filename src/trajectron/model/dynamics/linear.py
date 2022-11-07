from trajectron.model.dynamics import Dynamic


class Linear(Dynamic):
    def integrate_samples(self, v, x, dt):
        return v

    def integrate_distribution(self, v_dist, x, dt):
        return v_dist
