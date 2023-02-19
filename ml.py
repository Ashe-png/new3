import os
import sys
import numpy as np
import tensorflow as tf
import time
import yaml
import json
from tensorflow.keras.utils import Progbar
from model.dataset import Dataset
from model.fp.melspec.melspectrogram_tflite import get_melspec_layer
from model.fp.nnfp import get_fingerprinter
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from eval.utils.get_index_faiss import get_index
from eval.utils.print_table import PrintTable

def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_fp(cfg):
    """ Build fingerprinter """
    # m_pre: log-power-Mel-spectrogram layer, S.
    m_pre = get_melspec_layer(cfg, trainable=False)

    # m_fp: fingerprinter g(f(.)).
    m_fp = get_fingerprinter(cfg, trainable=False)
    return m_pre, m_fp

def load_checkpoint(checkpoint_root_dir, checkpoint_name, checkpoint_index,
                    m_fp):
    """ Load a trained fingerprinter """
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(model=m_fp)
    checkpoint_dir = checkpoint_root_dir + f'/{checkpoint_name}/'
    c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir,
                                           max_to_keep=None)

    # Load
    if checkpoint_index == None:
        tf.print("\x1b[1;32mArgument 'checkpoint_index' was not specified.\x1b[0m")
        tf.print('\x1b[1;32mSearching for the latest checkpoint...\x1b[0m')
        latest_checkpoint = c_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint_index = int(latest_checkpoint.split(sep='ckpt-')[-1])
            status = checkpoint.restore(latest_checkpoint)
            status.expect_partial()
            tf.print(f'---Restored from {c_manager.latest_checkpoint}---')
        else:
            raise FileNotFoundError(f'Cannot find checkpoint in {checkpoint_dir}')
    else:
        checkpoint_fpath = checkpoint_dir + 'ckpt-' + str(checkpoint_index)
        status = checkpoint.restore(checkpoint_fpath) # Let TF to handle error cases.
        status.expect_partial()
        tf.print(f'---Restored from {checkpoint_fpath}---')
    return checkpoint_index

def get_data_source(cfg):
    dataset = Dataset(cfg)
    ds = dict()

    # app = QApplication(sys.argv)

    # file_path, _ = QFileDialog.getOpenFileName()
    source_root_dir = './temp'
    # source_root_dir = os.path.relpath(file_path, os.getcwd())
    ds['custom_source'] = dataset.get_custom_db_ds(source_root_dir)
    return ds

# @tf.function
def test_step(X, m_pre, m_fp):
    """ Test step used for generating fingerprint """
    # X is not (Xa, Xp) here. The second element is reduced now.
    m_fp.trainable = False
    return m_fp(m_pre(X))  # (BSZ, Dim)

def generate_fingerprint(cfg):
    # Build and load checkpoint
    checkpoint_name = '640_lamb'
    checkpoint_index = 11
    m_pre, m_fp = build_fp(cfg)
    checkpoint_root_dir = cfg['DIR']['LOG_ROOT_DIR'] + 'checkpoint/'
    checkpoint_index = load_checkpoint(checkpoint_root_dir, checkpoint_name,
                                       checkpoint_index, m_fp)

    # Get data source
    """ ds = {'key1': <Dataset>, 'key2': <Dataset>, ...} """
    ds = get_data_source(cfg)

    # Generate
    sz_check = dict() # for warning message
    for key in ds.keys():
        bsz = int(cfg['BSZ']['TS_BATCH_SZ'])  # Do not use ds.bsz here.
        # n_items = len(ds[key]) * bsz
        n_items = ds[key].n_samples
        dim = cfg['MODEL']['EMB_SZ']

        # Create memmap, and save shapes
        assert n_items > 0
        arr_shape = (n_items, dim)
        
        arr = np.zeros(arr_shape, dtype=float)


        # Fingerprinting loop
        tf.print(
            f"=== Generating fingerprint from \x1b[1;32m'{key}'\x1b[0m " +
            f"bsz={bsz}, {n_items} items, d={dim}"+ " ===")
        progbar = Progbar(len(ds[key]))

        """ Parallelism to speed up preprocessing------------------------- """
        enq = tf.keras.utils.OrderedEnqueuer(ds[key],
                                              use_multiprocessing=True,
                                              shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                  max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            progbar.update(i)
            X, _ = next(enq.get())
            emb = test_step(X, m_pre, m_fp)
            arr[i * bsz:(i + 1) * bsz, :] = emb.numpy() # Writing on disk.
            i += 1
        progbar.update(i, finalize=True)
        enq.stop()
        """ End of Parallelism-------------------------------------------- """

        # tf.print(f'=== Succesfully stored {arr_shape[0]} fingerprint to {output_root_dir} ===')
        sz_check[key] = len(arr)
        return arr, arr_shape

def load_memmap_data(source_dir,
                     fname,
                     append_extra_length=None,
                     shape_only=False,
                     display=True):
    """
    Load data and datashape from the file path.
    • Get shape from [source_dir/fname_shape.npy}.
    • Load memmap data from [source_dir/fname.mm].
    Parameters
    ----------
    source_dir : (str)
    fname : (str)
        File name except extension.
    append_empty_length : None or (int)
        Length to appened empty vector when loading memmap. If activate, the
        file will be opened as 'r+' mode.
    shape_only : (bool), optional
        Return only shape. The default is False.
    display : (bool), optional
        The default is True.
    Returns
    -------
    (data, data_shape)
    """
    path_shape = source_dir + fname + '_shape.npy'
    path_data = source_dir + fname + '.mm'
    data_shape = np.load(path_shape)
    if shape_only:
        return data_shape

    if append_extra_length:
        data_shape[0] += append_extra_length
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    else:
        data = np.memmap(path_data, dtype='float32', mode='r',
                         shape=(data_shape[0], data_shape[1]))
    if display:
        print(f'Load {data_shape[0]:,} items from \033[32m{path_data}\033[0m.')
    return data, data_shape

def search(   db,
               db_shape,
               index,
               index_type='ivfpq',
               nogpu=True,
               max_train=1e7,
               test_ids='icassp',
               test_seq_len='1 3 5 9 11 19',
               k_probe=20,
               
               display_interval=5):
    """
    Segment/sequence-wise audio search experiment and evaluation: implementation based on FAISS.
    ex) python eval.py EMB_DIR --index_type ivfpq
    EMB_DIR: Directory where {query, db, dummy_db}.mm files are located. The 'raw_score.npy' and 'test_ids.npy' will be also created in the same directory.
    """
    # test_seq_len = np.asarray(
        # list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]

    # Load items from {query, db, dummy_db}
    # query, query_shape = load_memmap_data(emb_dir, 'query')
    emb_dir = './logs/emb/640_lamb/11/'

    # db, db_shape = load_memmap_data(emb_dir, 'custom_source')
    config = '640_lamb'
    emb_dummy_dir = emb_dir
    
    # dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')
    """ ----------------------------------------------------------------------
    FAISS index setup
        dummy: 10 items.
        db: 5 items.
        query: 5 items, corresponding to 'db'.
        index.add(dummy_db); index.add(db) # 'dummy_db' first
               |------ dummy_db ------|
        index: [d0, d1, d2,..., d8, d9, d11, d12, d13, d14, d15]
                                       |--------- db ----------|
                                       |--------query ---------|
                                       [q0,  q1,  q2,  q3,  q4]
    • The set of ground truth IDs for q[i] will be (i + len(dummy_db))
    ---------------------------------------------------------------------- """
    # # Create and train FAISS index
    # index = get_index(index_type, db, db.shape, (not nogpu),
    #                   max_train, trained=True)
    

    # # Add items to index
    # start_time = time.time()

    # # index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
    # index.add(db); print(f'{len(db)} items from reference DB')

    # t = time.time() - start_time
    # print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')

    """ ----------------------------------------------------------------------
    We need to prepare a merged {dummy_db + db} memmap:
    • Calcuation of sequence-level matching score requires reconstruction of
      vectors from FAISS index.
    • Unforunately, current faiss.index.reconstruct_n(id_start, id_stop)
      supports only CPU index.
    • We prepare a fake_recon_index thourgh the on-disk method.
    ---------------------------------------------------------------------- """
    # Prepare fake_recon_index
    # del dummy_db
    start_time = time.time()

    fake_recon_index, index_shape = load_memmap_data(
        emb_dummy_dir, 'custom_source',
        display=False)
    # fake_recon_index[dummy_db_shape[0]:dummy_db_shape[0] + query_shape[0], :] = db[:, :]
    # fake_recon_index.flush()

    t = time.time() - start_time
    print(f'Created fake_recon_index, total {index_shape[0]} items. {t:>4.2f} sec.')

    # # Get test_ids
    # print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')
    # if test_ids.lower() == 'all':
    #     test_ids = np.arange(0, len(query) - max(test_seq_len), 1) # will test all segments in query/db set
    # elif test_ids.lower() == 'icassp':
    #     test_ids = np.load(
    #         glob.glob('./**/test_ids_icassp2021.npy', recursive=True)[0])
    # elif test_ids.isnumeric():
    #     test_ids = np.random.permutation(len(query) - max(test_seq_len))[:int(test_ids)]
    # else:
    #     test_ids = np.load(test_ids)

    # n_test = len(test_ids)
    # gt_ids  = test_ids + dummy_db_shape[0]
    # print(f'n_test: \033[93m{n_test:n}\033[0m')

    """ Segement/sequence-level search & evaluation """
    start_time = time.time()
    cfg = load_config(config)
    query, query_shape = generate_fingerprint(cfg)
    q = query[:, :] # shape(q) = (length, dim)

    # segment-level top k search for each segment
    _, I = index.search(
        query, k_probe) # _: distance, I: result IDs matrix

    # offset compensation to get the start IDs of candidate sequences
    for offset in range(len(I)):
        I[offset, :] -= offset

    # unique candidates
    candidates = np.unique(I[np.where(I >= 0)])   # ignore id < 0

    """ Sequence match score """
    _scores = np.zeros(len(candidates))
    for ci, cid in enumerate(candidates):
        _scores[ci] = np.mean(
            np.diag(
                # np.dot(q, index.reconstruct_n(cid, (cid + l)).T)
                np.dot(q, fake_recon_index[cid:cid + 19, :].T)
                )
            )

    pred_id = candidates[np.argmax(_scores)] # <-- only top1-hit
    print(pred_id)
    return pred_id


def result(pred_id):

    with open('./eval/metadata.json', 'r') as f:
        data = json.load(f)
    column_data = [row['indices'] for row in data]

    filtered_list = [x for x in column_data if x <= pred_id]
    closest = min(filtered_list, key=lambda x: abs(x - pred_id))

    for song in data:
        if song['indices'] == closest:
            print (song)
            return song
    