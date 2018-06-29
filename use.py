import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition.nmf import non_negative_factorization
from numpy.linalg import norm
from optimal_transport import compute_permutation
import librosa

ins = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead',
       'vocal']

abbrevs = {'length': 'l', 'layers': 'lyr', 'n_components': 'cpn', 'examples': 'ex', 'epochs': 'ep', 'qualities': 'qult',
           'lambd': 'lbd'}


def gt_s_path(suppath, **kwargs):
    path = ''
    for name, value in kwargs.items():
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

        elif not name.endswith(('dir', 'path')) and value is not None:
            if name in abbrevs.keys():
                name = abbrevs[name]
            if isinstance(value, (list, tuple)):
                vals = ''
                for i in value:
                    vals += '-%d' % i
                value = vals
            path += '{}-{}_'.format(name, value)

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


def compare_2_matrix(ws, wt, figdir):
    figs, axs = plt.subplots(1, 2, figsize=(40, 10))
    axs[0].set_aspect('equal')
    axs[0].imshow(ws, interpolation='nearest', cmap=plt.cm.ocean)
    axs[1].set_aspect('equal')
    axs[1].imshow(wt, interpolation='nearest', cmap=plt.cm.ocean)
    #plt.colorbar()
    plt.savefig(os.path.join(figdir, 'ws-wt.png'), dpi=50)


def transform(enc, ws, wt, n_components, figdir=None):
    enc = enc[0]
    hT, _, _ = non_negative_factorization(enc, n_components=n_components, H=ws.T, update_H=False,
                                          solver='mu', max_iter=400, verbose=1)
    wt = compute_permutation(ws, wt)

    if figdir is not None:
        compare_2_matrix(ws, wt, figdir)

    u = np.matmul(hT, ws.T)
    print(' Error for ws * h_ = enc: {}'.format(norm(enc - u) / norm(enc)))

    return np.expand_dims(np.matmul(hT, wt.T), axis=0)


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


