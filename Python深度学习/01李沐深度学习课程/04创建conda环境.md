```
Windows PowerShell
版权所有 (C) Microsoft Corporation。保留所有权利。

尝试新的跨平台 PowerShell https://aka.ms/pscore6

PS C:\Users\hp> conda --version
conda 4.9.2
PS C:\Users\hp> conda create --name d2l python=3.8 -y
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.9.2
  latest version: 4.13.0

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: I:\miniconda\envs\d2l

  added / updated specs:
    - python=3.8


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2022.4.26  |       h9f7ea03_0         124 KB
    certifi-2022.6.15          |   py38h9f7ea03_0         162 KB
    openssl-1.1.1o             |       hc431981_0         4.4 MB
    pip-21.2.4                 |   py38h9f7ea03_0         1.8 MB
done
#
# To activate this environment, use
#
#     $ conda activate d2l
#
# To deactivate an active environment, use
#
#     $ conda deactivate

PS C:\Users\hp>
```

激活环境
```
conda activate d2l
```

取消激活环境
```
conda deactivate
```

安装pytorch
```
(d2l) C:\Users\hp>pip --version
pip 21.2.4 from I:\miniconda\envs\d2l\lib\site-packages\pip (python 3.8)

(d2l) C:\Users\hp>conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

PackagesNotFoundError: The following packages are not available from current channels:

  - torchvision==0.13.0
  - pytorch==1.12.0
  - torchaudio==0.12.0
  - cudatoolkit=11.3

Current channels:

  - https://conda.anaconda.org/pytorch/win-32
  - https://conda.anaconda.org/pytorch/noarch
  - https://repo.anaconda.com/pkgs/main/win-32
  - https://repo.anaconda.com/pkgs/main/noarch
  - https://repo.anaconda.com/pkgs/r/win-32
  - https://repo.anaconda.com/pkgs/r/noarch
  - https://repo.anaconda.com/pkgs/msys2/win-32
  - https://repo.anaconda.com/pkgs/msys2/noarch

To search for alternate channels that may provide the conda package you're
looking for, navigate to

    https://anaconda.org

and use the search bar at the top of the page.
```