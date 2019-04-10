import tensorflow as tf
import numpy as np
import threading
import time


# 定义每个线程执行的操作
def thread_op(coordinator, thread_id):
    # 判断should_stop()函数的状态
    while coordinator.should_stop() is False:
        if np.random.rand() < 0.1:
            print("Stoping from thread_id: {}".format(thread_id))
            # 调用request_stop()函数请求所有线程停止
            # request_stop(self,ex)
            coordinator.request_stop()
        else:
            print("Working on thread_id:{}".format(thread_id))

        # 如果线程没有停止，则休息10秒钟后再次执行循环
        time.sleep(2)


# 实例化Coordinator类
coordinator_t = tf.train.Coordinator()

# 通过Python中threading类的Thread()函数创建5个线程
# __init__(self,group,target,name,args,kwargs,daemon)
threads = [threading.Thread(target=thread_op, args=(coordinator_t, i)) for i in range(5)]

# 启动创建的5个线程
for j in threads:
    # start(self)
    j.start()

# 将Coordinator类加入到线程并等待所有线程退出
# join(self,threads,stop_grace_period_secs)
coordinator_t.join(threads)
