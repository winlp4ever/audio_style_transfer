### Audio Style Transfer

The project is in `python3.x`
Required packages:
* tensorflow >=1.8
* librosa
* numpy
* scipy
* matplotlib
* sklearn

---
#### How to run?
The main file is `methods.py`. Type `python methods.py --help` to get more details.
Let's take an example:

```bash
python methods.py pachelbel organ --epochs 100 --cont_lyrs 25 --stack 0 --lambd 100 --gamma 0
``` 

This code will take the file `<srcdir>/pachelbel.wav` and try to transfer its style to the file `<srcdir>/organ.wav`'s style. 

`<srcdir>` is set default to `./data/src`. You should copy all your reference *content* and *style* files to this dir. You can change `<srcdir>` by adding `--dir <where-you-store-your-src-files>` to the above code line.

You can also change the start and duration of your content file (the output will be of same offset and duration) via args `--start` (default to `1`) and `--batch_size` (default to `16384`), 
that way your output will start at second `1` and last `batch-size/sampling-rate` second(s).

The method is by default ours (*channel-wise gram matrices*), you can switch to
**Gatys method** by adding the argument `--gatys` to the above code line.

Again, type `--help` for more information.

Besides, in this folder you can also find `spectrogram.py` and `rainbowgram.py` which serve to print out spectrograms and cqtgrams  (constant-Q grams like in NSynth's paper) respectively of a signal.

---
#### Use my environment
You can either create a new environment or add the following function to your `~/.bashrc`
```bash
function enableEnv() {
    # param = no gpu : 0 .. 7
    thegpuu=${1:-0}
    export ANACONDA=/home/wp01/interns/leh_2018/anaconda3
    export PATH="$ANACONDA/bin:$PATH"
    export PATH="/usr/local/cuda/bin:$PATH"
    export CUDA_ROOT=/usr/local/cuda-8.0/
    export PATH=$CUDA_ROOT/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:/usr/lib64/atlas:$LD_LIBRARY_PATH:$$
    export CUDA_VISIBLE_DEVICES=${thegpu}
    export THEANO_FLAGS=nvcc.flags=-arch=sm_50,cuda.root=$CUDA_ROOT,device=gpu0$
    LD=/usr/local/lib:/usr/local4/lib:/usr/local/lib:/usr/local/lib:/usr/local2$
    export LD_LIBRARY_PATH=$LD
    export PATH="$PATH:/usr/local/cuda-8.0/bin"
    export leh=/home/wp01/interns/leh_2018
}
```

To activate the env., simply type
```bash
source ~/.bashrc # if you're on a server, you need to source your bashrc first
enableEnv
source activate tensorflow
```

or you can just simply write a `bash script` like follows and source it
every time you log in to a machine
```bash
#! /usr/bin/env bash
source ~/.bashrc
enableEnv
source activate tensorflow
for i in $( seq 1 4 ); do
        if [ -z ${!i} ]; then
                break
        elif [ -z $gpus ]; then
                gpus=${!i}
        else
                gpus=$gpus,${!i}
        fi
done
if [ ! -z $gpus ]; then
        export CUDA_VISIBLE_DEVICES=$gpus
        IFS=',' read -r -a arr <<< "$gpus"
        if [ ${#arr[@]} -gt 1 ]; then
                echo "registered gpu devices are $gpus .OK!"
        else
                echo "registered gpu device is $gpus .OK!"
        fi
fi
cd $leh/audio_style_transfer # project dir
```

For example, store it at `~/runleh.sh`

each time you want to lance the env, just type
```bash
source ~/runleh.sh
```

If you type
```bash
source ~/runleh.sh 2 4
```
then your program will automatically register `gpu` 2 and 4 for your program

