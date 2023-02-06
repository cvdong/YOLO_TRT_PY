'''
Version: v1.0
Author: 東DONG
Mail: cv_yang@126.com
Date: 2022-11-01 08:11:39
LastEditTime: 2023-02-06 15:55:52
FilePath: /YOLO_TRT_PY/src/Inference_trt_det.py
Description: 
Copyright (c) 2022 by ${東}, All Rights Reserved. 

                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='

     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑     永不宕机     永无BUG

'''

import os
import tensorrt as trt
import pycuda.driver as cuda 
import pycuda.autoinit
import numpy as np
import cv2
import argparse
import time 
import multiprocessing as mp

ctx = pycuda.autoinit.context
trt.init_libnvinfer_plugins(None, "")

Parser = argparse.ArgumentParser()
Parser.add_argument('--engine_path', type=str, default='workspace/yolov5s_fp16.engine')
Parser.add_argument('--image_path', type=str, default='workspace/inference')
Parser.add_argument('--result_path', type=str, default='workspace/result')
args = Parser.parse_args()


# 构建trt推理接口
class InferenceTrt():
  
    def __init__(self, engine, context):
        
        self.inputs = []
        self.outputs = []
        self.engine = engine
        self.context = context
              
    # 反序列化engine
    @staticmethod
    def param_set(engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context() 
        
        return engine, context
    
    # 前处理
    def pre_process(self, image, input_w=640, input_h=640):
        scale = min(input_h / image.shape[0], input_w / image.shape[1])
        ox = (-scale * image.shape[1] + input_w + scale  - 1) * 0.5
        oy = (-scale * image.shape[0] + input_h + scale  - 1) * 0.5
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)
        IM = cv2.invertAffineTransform(M)

        image_prep = cv2.warpAffine(image, M, (input_w, input_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        image_prep = (image_prep[..., ::-1] / 255.0).astype(np.float32)
        image_prep = image_prep.transpose(2, 0, 1)[None]
        return image_prep, M, IM
            
    # NMS
    def nms(self, boxes, threshold=0.5):

        keep = []
        remove_flags = [False] * len(boxes)
        for i in range(len(boxes)):

            if remove_flags[i]:
                continue

            ib = boxes[i]
            keep.append(ib)
            for j in range(len(boxes)):
                if remove_flags[j]:
                    continue

                jb = boxes[j]

                # class mismatch or image_id mismatch
                if ib[6] != jb[6] or ib[5] != jb[5]:
                    continue

                cleft,  ctop    = max(ib[:2], jb[:2])
                cright, cbottom = min(ib[2:4], jb[2:4])
                cross = max(0, cright - cleft) * max(0, cbottom - ctop)
                union = max(0, ib[2] - ib[0]) * max(0, ib[3] - ib[1]) + max(0, jb[2] - jb[0]) * max(0, jb[3] - jb[1]) - cross
                iou = cross / union
                if iou >= threshold:
                    remove_flags[j] = True
        return keep
    
    # 后处理
    def post_process(self, pred, IM, threshold=0.25):

        # b, n, 85
        boxes = []
        for image_id, box_id in zip(*np.where(pred[..., 4] >= threshold)):
            item = pred[image_id, box_id]
            cx, cy, w, h, objness = item[:5]
            label = item[5:].argmax()
            confidence = item[5 + label] * objness
            if confidence < threshold:
                continue

            boxes.append([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5, confidence, image_id, label])
        
        boxes = np.array(boxes)
        
        if  len(boxes):
            lr = boxes[:, [0, 2]]
            tb = boxes[:, [1, 3]]
            boxes[:, [0, 2]] = lr * IM[0, 0] + IM[0, 2]
            boxes[:, [1, 3]] = tb * IM[1, 1] + IM[1, 2]

            # left, top, right, bottom, confidence, image_id, label
            boxes = sorted(boxes.tolist(), key=lambda x:x[4], reverse=True)
            return self.nms(boxes)
        else:
            
            return self.nms(boxes)
        
    # 分配内存
    def allocate_buffers(self):
        
        bindings = []
        
        for binding in self.engine:
            
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_memory = cuda.pagelocked_empty(size, dtype)
            device_memory = cuda.mem_alloc(host_memory.nbytes)
            bindings.append(int(device_memory))
            
            if self.engine.binding_is_input(binding):
                
                self.inputs.append({'host':host_memory, 'device':device_memory})
            else:
                
                 self.outputs.append({'host':host_memory, 'device':device_memory})
        
        return bindings
    
    # trt推理
    def inference_trt(self, image):

        image, M, IM = self.pre_process(image)
       
        bindings_buffers = self.allocate_buffers()
        
        self.inputs[0]['host'] = np.ravel(np.ascontiguousarray(image))
        
        stream = cuda.Stream()
        
        for inp in self.inputs:
            # host-->device
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        
        self.context.execute_async_v2(bindings=bindings_buffers, stream_handle=stream.handle)
        
        for out in self.outputs:
            # device-->host
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)  
            
        # 流同步
        stream.synchronize()  
        
        output_shape = self.engine.get_binding_shape(1)
        res_trt = np.reshape(out['host'], output_shape)
        
        outputs = self.post_process(res_trt, IM) 
        
        return outputs
    
    # 清理缓存
    def clear_buff(self):
            
        self.inputs = []
        self.outputs = []
        
    # 推理时间
    def inference_test_trt(self, image):
        
        image, M, IM = self.pre_process(image)
        
        bindings_buffers = self.allocate_buffers()
        
        self.inputs[0]['host'] = np.ravel(np.ascontiguousarray(image))
        
        stream = cuda.Stream()
        
        for inp in self.inputs:
            # host-->device
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            
        # 计算推理时间
        start_time = time.time()
        
        for i in range(10000):
            self.context.execute_async_v2(bindings=bindings_buffers, stream_handle=stream.handle)
            i = i+1
            
            print(f'------------yolov5 tensorrt inference runing {i}--------------')
            
        end_time = time.time()
        
        for out in self.outputs:
            # device-->host
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)  
            
        # 流同步
        stream.synchronize()  
        
        output_shape = self.engine.get_binding_shape(1)
        res_trt = np.reshape(out['host'], output_shape)
        
        outputs = self.post_process(res_trt, IM) 
        
        spend_time = end_time-start_time
        
        print("----------Tensorrt 推理完成!!! 模型推理时间为{:.2f}ms-----------".format(spend_time/10))
        
        return outputs
    
# 颜色选择
class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
  
# 画框
class DrawResults():
         
    def __init__(self, det_results, classes_name, color):
        
        self.det_results = det_results
        self.classes_name = classes_name
        self.color = color
        
    def draw_det(self, image):
                        
        for obj in self.det_results:
           
            left, top, right, bottom = map(int, obj[:4])
            confidence = obj[4]
            label = self.classes_name[int(obj[6])]
        
            label_size = cv2.getTextSize(label + '00000', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (left, top-label_size[1]-3), (left + label_size[0], top-3), self.color(int(obj[6])), -1)
            
            cv2.rectangle(image, (left, top), (right, bottom), self.color(int(obj[6]), True), 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (left, top-3), 0, 0.5, (0, 0, 0), 2, 8)
            
            cv2.line(image, (left, top), (left+16, top), self.color(int(obj[6])+6,True), 2)
            cv2.line(image, (left, top), (left, top+16), self.color(int(obj[6])+6,True), 2)
            
            cv2.line(image, (right, top), (right-16, top), self.color(int(obj[6])+6,True), 2)
            cv2.line(image, (right, top), (right, top+16), self.color(int(obj[6])+6,True), 2)
            
            cv2.line(image, (left, bottom), (left+16, bottom), self.color(int(obj[6])+6,True), 2)
            cv2.line(image, (left, bottom), (left, bottom-16), self.color(int(obj[6])+6,True), 2)
            
            cv2.line(image, (right, bottom), (right-16, bottom), self.color(int(obj[6])+6,True), 2)
            cv2.line(image, (right, bottom), (right, bottom-16), self.color(int(obj[6])+6,True), 2)
              
        return image

# mp多进程
class TrtMp():
        
    def __init__(self, camera_ip_l, classes_name, color):
        
        self.camera_ip_l = camera_ip_l
        self.classes_name = classes_name
        self.color = color
        
    def image_put(self, q, camera_ip):
          
        cap = cv2.VideoCapture(camera_ip)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if cap.isOpened():
            print('----------got video_stream--------------')

        while cap.isOpened():
            
            ret, frame = cap.read() 
            if not ret:
                break
            
            q.put(frame)   
            q.get() if q.qsize() > 1 else time.sleep(1/fps)
        
        cap.release() 
        
    def image_get(self, q, window_name):
        
        engine, context= InferenceTrt.param_set(args.engine_path)  
        infer= InferenceTrt(engine, context)
    
        while True:
            frame = q.get()
            
            outputs = infer.inference_trt(frame)
            draw_fun = DrawResults(outputs, self.classes_name, self.color)
            draw_det = draw_fun.draw_det(frame)
            cv2.imshow(window_name, draw_det)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            infer.clear_buff()    
            
    def run_multi_camera(self):
        
        mp.set_start_method(method='spawn')  # init
        queues = [mp.Queue(maxsize=0) for _ in self.camera_ip_l]
        processes = []
    
        for queue, camera_ip in zip(queues, self.camera_ip_l):
             
            processes.append(mp.Process(target=self.image_put, args=(queue, camera_ip,)))
            processes.append(mp.Process(target=self.image_get, args=(queue, camera_ip,)))

        for process in processes:
            process.daemon = True
            process.start()
            
        for process in processes:
            process.join()

# inference image
def inference_image():
    
    classes_name = list(map(lambda x:x.strip(), open('workspace/coco.names', 'r').readlines()))
    color = Colors()
    
    # 读入图像
    image_paths = os.listdir(args.image_path)
    
    for image_path in image_paths:
        image = cv2.imread(args.image_path + '/' + image_path)
        
        engine, context= InferenceTrt.param_set(args.engine_path)
        # infer 接口实例化
        infer= InferenceTrt(engine, context)
        # 推理
        outputs = infer.inference_trt(image)
        
        # 推理耗时
        # outputs = infer.inference_test_trt(image)
        
        # 可视化
        draw_fun = DrawResults(outputs, classes_name, color)
        draw_det = draw_fun.draw_det(image)
        
        cv2.imwrite(args.result_path + '/' + image_path, image)
        cv2.namedWindow("inference", 0)
        cv2.imshow("inference", draw_det)
        
        cv2.waitKey(0)
    
# inference viedeo  
def inference_video():
    classes_name = list(map(lambda x:x.strip(), open('workspace/coco.names', 'r').readlines()))
    color = Colors()
    
    # 读入图像
    cap = cv2.VideoCapture("workspace/vtest.avi")
    engine, context= InferenceTrt.param_set(args.engine_path)
    # infer 接口实例化
    infer= InferenceTrt(engine, context)
    
    while cap.isOpened():
        
        ret, frame = cap.read() 
        if not ret:
            break
        
        # 推理
        outputs = infer.inference_trt(frame)
        # 可视化
        draw_fun = DrawResults(outputs, classes_name, color)
        draw_det = draw_fun.draw_det(frame)
       
        cv2.imshow("inference", draw_det)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        infer.clear_buff()  
    
    cv2.destroyAllWindows()
    
        
# inference videmo mp  
def inference_video_mp():
    
    classes = list(map(lambda x:x.strip(), open('workspace/coco.names', 'r').readlines()))
    color = Colors()
    
    camera_ip_l = [
        "workspace/vtest.avi", 
        "workspace/vtest.avi",   
        "workspace/vtest.avi", 
        ]
    
    mp_run = TrtMp(camera_ip_l, classes, color)
    mp_run.run_multi_camera()

  
if __name__=='__main__':
    
    inference_image()
    # inference_video()
    # inference_video_mp()