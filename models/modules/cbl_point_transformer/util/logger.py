
import os, sys, time, psutil, traceback
import subprocess as sp
import numpy as np

def print_mem(prefix, gpu=True, check_time=False, check_sys=False, **kwargs):
    sep = '\n\t' if any([gpu, check_time]) else ' '
    lines = [prefix, 'Mem Comsumption: %.2f GB' % (print_mem.process.memory_info()[0] / float(2**30))]
    if check_sys:
        sysmem = psutil.virtual_memory()
        lines += [f'Mem in sys: avail {sysmem.available / 2**30:.2f} / total {sysmem.total / 2**30:.2f}']
    if gpu:
        try:
            gpu_mem = get_gpu_mem()
            lines += [f'Availabel Mem of each GPU: {gpu_mem}']
        except FileNotFoundError:
            pass
        except sp.CalledProcessError:
            pass
    if check_time:
        cur_t = time.time()
        if not hasattr(print_mem, 't_start'):
            print_mem.t_start = cur_t
            print_mem.t = cur_t
        else:
            gap = int(cur_t-print_mem.t)
            cum = int(cur_t-print_mem.t_start)
            lines += [f'time used [gap/cum] : {gap // 60}min {gap % 60}s / {cum // 60}min {cum % 60}s']
            print_mem.t = cur_t
    print(sep.join(lines), **kwargs)
print_mem.process = psutil.Process(os.getpid())


def get_gpu_mem():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def print_dict(d, prefix='', except_k=[], fn=None, head=None, sort=False):
    if head is not None:
        d = {head: d}

    kwargs = {  # those need to be passed into recusive calls
        'except_k': except_k,
        'fn': fn,
        'sort': sort,
    }

    k_list = list(d.keys())
    if sort:
        k_list = sorted(k_list)
    for k in k_list:
        v = d[k]
        if k in except_k:
            continue
        if isinstance(d[k], dict):
            print(f'{prefix}{str(k)}:')
            print_dict(d[k], prefix=f'{prefix}\t', **kwargs)
        else:
            if fn:
                rst = None
                try:
                    if isinstance(v, (list, tuple)):
                        rst = v.__class__([fn(vv) for vv in v])
                    else:
                        rst = fn(v)
                except:
                    pass
                v = rst if rst is not None else v
            line = f'{prefix}{str(k)}\t{str(v)}'
            if isinstance(v, (list, tuple)) and len(str(line)) > 100:  # overlong
                line_pre = f'{prefix}{str(k)}\t' + ('[' if isinstance(v, list) else '(')
                line_post = f'\n{prefix}\t' + (']' if isinstance(v, list) else ')')
                if set([type(s) for s in v]) == set([dict]):  # all dict in list
                    print(line_pre)
                    for s in v[:-1]:
                        print_dict(s, prefix=f'{prefix}\t\t', **kwargs)
                        print(f'{prefix}\t\t,')
                    print_dict(v[-1], prefix=f'{prefix}\t\t', **kwargs)
                    line = line_post
                else:
                    line =  line_pre + f'\n{prefix}\t\t'.join([''] + [str(s) for s in v]) + line_post

            print(line)
    print(flush=True)
