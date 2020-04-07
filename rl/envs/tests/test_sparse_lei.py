from rl.envs.sparse_lei import SparseLei


def test_class_sparse_lei():
    print("")

    env = SparseLei()
    env.seed(1)
    env.play()

    env = SparseLei(nuisance=7, tau=1.0, shift=0.0, noise=0.01)
    env.seed(1)
    env.play()
