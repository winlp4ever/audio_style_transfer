import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition.nmf import non_negative_factorization
from numpy.linalg import norm
from optimal_transport import compute_permutation
import librosa
import tensorflow as tf

ins = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead',
       'vocal']

abbrevs = {'length': 'l', 'layers': 'lyr', 'n_components': 'cpn', 'examples': 'ex', 'epochs': 'ep', 'qualities': 'qult',
           'lambd': 'lbd', 'batch_size': 'btch', 'stack': 'stk'}


def gt_s_path(suppath, exe_file=None, **kwargs):
    path = ''
    for name, value in sorted(kwargs.items()):
        if name == 'ins' and value is not None:
            assert len(value) == 2
            path += '{}2{}_'.format(ins[value[0]], ins[value[1]])

        elif name == 'male2female':
            assert value <= 2
            if value == 0:
                path += 'f2m_'
            elif value == 1:
                path += 'm2f_'

        elif name == 'filename':
            path = value + '_' + path

        elif name == 'cont_fn' or name == 'style_fn':
            path += '-{}-'.format(value)

        elif not name.endswith(('dir', 'path')) and value is not None:
            if name in abbrevs.keys():
                name = abbrevs[name]
            if isinstance(value, (list, tuple)):
                vals = ''
                for i in value:
                    vals += '-%d' % i
                value = vals
            path += '{}-{}_'.format(name, value)
    if exe_file:
        path = exe_file + '::' + path
    path = os.path.join(suppath, path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def crt_t_fol(suppath, hour=False):
    dte = time.localtime()
    if hour:
        fol_n = os.path.join(suppath, '{}{}{}{}'.format(dte[1], dte[2], dte[3], dte[4]))
    else:
        fol_n = os.path.join(suppath, '{}{}'.format(dte[1], dte[2]))

    if not os.path.exists(fol_n):
        os.makedirs(fol_n)
    return fol_n


def mu_law_numpy(x, mu=255):
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    out = np.floor(out * 128)
    return out


def inv_mu_law_numpy(x, mu=255.0):
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu) ** np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out

def abs(x):
    return tf.maximum(x, 1e-12) + tf.maximum(0.0, -x)

def sign(x):
    out = tf.where(tf.less_equal(tf.abs(x), 1e-12), tf.zeros_like(x), x)
    return out / abs(x)

def inv_mu_law(x, mu=255):
    x = tf.cast(x, tf.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = sign(out) / mu * ((1 + mu) ** abs(out) - 1)
    out = tf.where(tf.equal(x, 0), x, out)
    return out


def compare_2_matrix(ws, wt, figdir, save_matrices=False):
    figs, axs = plt.subplots(1, 2, figsize=(10, 40))
    axs[0].set_aspect('equal')
    im0 = axs[0].imshow(ws, interpolation='nearest', cmap=plt.cm.ocean)
    axs[1].set_aspect('equal')
    im1 = axs[1].imshow(wt, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar(im0, ax=axs[0])
    plt.colorbar(im1, ax=axs[1])
    plt.savefig(os.path.join(figdir, 'ws-wt.png'), dpi=50)

    rows, cols = ws.shape

    for i in range(cols):
        figs, axs = plt.subplots(1, 2, figsize=(20, 5))
        axs[0].plot(ws[:, i])
        axs[0].set_ylim(top=1.)
        axs[1].plot(wt[:, i])
        axs[1].set_ylim(top=1.)
        plt.savefig(os.path.join(figdir, 'ws-wt-col{}.png'.format(i)), dpi=50)

    np.save(os.path.join(figdir,'ws'), arr=ws)
    np.save(os.path.join(figdir,'wt'), arr=wt)
    plt.close()


def transform(enc, ws, wt, n_components, figdir=None):
    enc = enc[0]
    hT, _, _ = non_negative_factorization(enc, n_components=n_components, H=ws.T, update_H=False,
                                          solver='mu', max_iter=400, verbose=1)
    wt = compute_permutation(ws, wt)

    if figdir is not None:
        compare_2_matrix(ws, wt, figdir)

    u = np.matmul(hT, ws.T)
    print(' Error for ws * h_ = enc: {}'.format(norm(enc - u) / norm(enc)))
    print(' difference between two matrices {}'.format(norm(ws - wt) / norm(ws)))

    return np.expand_dims(np.matmul(hT, ws.T), axis=0)


def vis_actis(aud, enc, fig_dir, ep, layers, nb_channels=5, dspl=64, output_file=False):
    nb_layers = enc.shape[0]
    fig, axs = plt.subplots(nb_layers + 1, 3, figsize=(30, 5 * (nb_layers + 1)))
    axs[0, 1].plot(aud)
    axs[0, 1].set_title('Audio Signal')
    axs[0, 0].axis('off')
    axs[0, 2].axis('off')
    for i in range(nb_layers):
        axs[i + 1, 0].plot(np.log(enc[i, :dspl, :nb_channels] + 1))
        axs[i + 1, 0].set_title('Embeds layer {} part 0'.format(layers[i]))
        axs[i + 1, 1].plot(np.log(enc[i, dspl:2 * dspl, :nb_channels] + 1))
        axs[i + 1, 1].set_title('Embeds layer {} part 1'.format(layers[i]))
        axs[i + 1, 2].plot(np.log(enc[i, 2 * dspl:3 * dspl, :nb_channels] + 1))
        axs[i + 1, 2].set_title('Embeds layer {} part 2'.format(layers[i]))
    plt.savefig(os.path.join(fig_dir, 'f-{}.png'.format(ep)), dpi=50)
    sp = os.path.join(fig_dir, 'f-{}'.format(ep))

    plt.savefig(sp + '.png', dpi=50)
    if output_file:
        librosa.output.write_wav(sp + '.wav', aud, sr=16000)


def vis_actis_ens(aud, enc, fig_dir, ep, layer_ids, nb_channels=5, dspl=256, output_file=False):
    nb_layers = enc.shape[0]
    fig, axs = plt.subplots(nb_layers + 1, 3, figsize=(30, 5 * (nb_layers + 1)))
    axs[0, 1].plot(aud)
    axs[0, 1].set_title('Audio Signal')
    axs[0, 0].axis('off')
    axs[0, 2].axis('off')

    for i in range(nb_layers):
        a = np.reshape(enc[i, :, :nb_channels], [-1, dspl, nb_channels])
        std = np.std(a, axis=1)
        mean = np.mean(a, axis=1)
        min = np.std(a, axis=1)
        max = np.std(a, axis=1)
        axs[i + 1, 0].plot(min)
        axs[i + 1, 0].plot(max)
        axs[i + 1, 0].set_title('embeds layer {} -- MIN/MAX'.format(layer_ids[i]))
        axs[i + 1, 1].plot(std + mean)
        axs[i + 1, 1].plot(-std + mean)
        axs[i + 1, 1].set_title('embeds layer {} -- STD/MEAN'.format(layer_ids[i]))
        axs[i + 1, 2].plot(mean)
        axs[i + 1, 2].set_title('embeds layer {} -- AVG'.format(layer_ids[i]))

    sp = os.path.join(fig_dir, 'fe-{}'.format(ep))
    plt.savefig(sp + '.png', dpi=50)
    if output_file:
        librosa.output.write_wav(sp + '.wav', aud, sr=16000)

def vis_mats(phis, phit, layer_ids, figdir=None, srcname=None, trgname=None):
    fig, axs = plt.subplots(len(layer_ids) + 1, 2, figsize=(40, 10 * len(layer_ids) + 1))
    if srcname:
        axs[0, 0].set_title(srcname)
    if trgname:
        axs[0, 1].set_title(trgname)
    axs[0, 0].imshow(phis, interpolation='nearest', cmap=plt.cm.plasma, aspect='auto')
    axs[0, 1].imshow(phit, interpolation='nearest', cmap=plt.cm.plasma, aspect='auto')
    for i in range(len(layer_ids)):
        ps = phis[i * 128:(i + 1) * 128]
        pt = phit[i * 128:(i + 1) * 128]

        ps = np.dot(ps, ps.T)
        pt = np.dot(pt, pt.T)
        axs[i + 1, 0].set_title('layer-{}'.format(layer_ids[i]))
        axs[i + 1, 0].imshow(ps / np.max(ps), interpolation='nearest', cmap=plt.cm.plasma)

        axs[i + 1, 1].set_title('layer-{}'.format(layer_ids[i]))
        im = axs[i + 1, 1].imshow(pt / np.max(pt), interpolation='nearest', cmap=plt.cm.plasma)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    print('save mat fig ...')
    if figdir:
        plt.savefig(os.path.join(figdir, 'mats_plt.png'), dpi=150)
    else:
        plt.show()
    plt.close()

def show_gram(mats, ep=None, figdir=None):
    nb_chnnls = mats.shape[0]
    fig, axs = plt.subplots(10, nb_chnnls // 10, figsize=(12 * nb_chnnls // 10, 100))
    for i in range(10):
        for j in range(nb_chnnls // 10):
            axs[i, j].imshow(mats[i + j * 10], interpolation='nearest', cmap=plt.cm.plasma)
            axs[i, j].set_title('channel {}'.format(i + 10 * j))
    if ep is not None:
        fig.savefig(os.path.join(figdir, 'gram-ep{}.png'.format(ep)), dpi=50)
    else:
        fig.savefig(os.path.join(figdir, 'gram-style.png'), dpi=50)
    plt.close()

def load_audio(fn, sr, audio_channel):
    audio, sr = librosa.load(fn, sr=sr, mono=False)
    if len(audio.shape) > 1:
        return audio[audio_channel], sr
    else:
        return audio, sr