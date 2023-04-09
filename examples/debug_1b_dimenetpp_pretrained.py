
    @classmethod
    def from_qm9_pretrained(
        cls,
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['DimeNetPlusPlus', Dataset, Dataset,
               Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`DimeNetPlusPlus` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        assert target >= 0 and target <= 12 and not target == 4

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet_pp', qm9_target_dict[target])

        makedirs(path)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        # Configuration from DimeNet++:
        # https://github.com/gasteigerjo/dimenet/blob/master/config_pp.yaml
        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name, transpose=False):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name[-6:] == 'kernel':
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_up.weight,
                  f'output_blocks/{i}/up_projection/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf1.weight, f'int_blocks/{i}/dense_rbf1/kernel')
            copy_(block.lin_rbf2.weight, f'int_blocks/{i}/dense_rbf2/kernel')
            copy_(block.lin_sbf1.weight, f'int_blocks/{i}/dense_sbf1/kernel')
            copy_(block.lin_sbf2.weight, f'int_blocks/{i}/dense_sbf2/kernel')

            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')

            copy_(block.lin_down.weight,
                  f'int_blocks/{i}/down_projection/kernel')
            copy_(block.lin_up.weight, f'int_blocks/{i}/up_projection/kernel')

            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')

            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')

            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        perm = perm.long()
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])
