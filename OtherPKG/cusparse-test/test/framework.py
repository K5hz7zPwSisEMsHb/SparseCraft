import time
import queue
from . import external_exec, mtx_dir_path

task_q = queue.Queue()
info_q = queue.Queue()
erro_q = queue.Queue()

def workload(task, mtx, process, *args):
    print(f'./build/{task} "{mtx_dir_path}/{mtx}" {" ".join([str(i) for i in args])}')
    st, rt = external_exec(f'./build/{task} "{mtx_dir_path}/{mtx}" {" ".join([str(i) for i in args])}', without_output=True)
    if not st:
        rt = rt.splitlines()[-1]
        info_q.put([mtx] + rt.strip().split(','))
    elif st != 1:
        erro_q.put([mtx, rt])
    process[0].advance(process[1])

def info_consumer(odf, csv):
    while True:
        info = info_q.get()
        if not info:
            break
        odf.loc[odf.shape[0]] = [odf.shape[0]] + info
        odf.to_csv(csv, index=False)

def erro_consumer(task):
    while True:
        erro = erro_q.get()
        if not erro:
            break
        with open(f'dist/odf/{task}/err.log', 'a') as f:
            f.write(f'{erro[0]}\n{erro[1]}\n\n')

def consumer(task, snum, ls):
    while ls:
        mtx = ls.pop(0)
        workload(task, mtx, snum)
    info_q.put(None)
    erro_q.put(None)

def start_framework(task, ls, odf, csv, *args):
    from concurrent.futures import ThreadPoolExecutor, wait
    
    executor = ThreadPoolExecutor(max_workers=2)
    init_thread_manage = []
    
    try:
        init_thread_manage.append(executor.submit(info_consumer, odf, csv))
        init_thread_manage.append(executor.submit(erro_consumer, task))
        
        from QuickStart_Rhy.TuiTools.Bar import NormalProgressBar

        process, task_id = NormalProgressBar(task, len(ls))
        process.start()
        while ls:
            mtx = ls.pop(0)
            workload(task, mtx, (process, task_id), *args)
            # thread_manage.append(executor.submit(workload, task, mtx, process=(process, task_id)))

        # wait(thread_manage)
        info_q.put(None)
        erro_q.put(None)
        if process:
            process.stop()
        wait(init_thread_manage)
    except KeyboardInterrupt:
        info_q.put(None)
        erro_q.put(None)
        if process:
            process.stop()
        wait(init_thread_manage)
