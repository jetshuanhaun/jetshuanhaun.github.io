import{_ as s,o as n,c as a,f as l}from"./app.a56962f7.js";const C=JSON.parse('{"title":"Yolov7进行人体关节点检测","description":"使用Yolov7来检测人体关节点","frontmatter":{"title":"Yolov7进行人体关节点检测","tags":["Yolov7"],"layout":"post","date":"2023-06-30","description":"使用Yolov7来检测人体关节点"},"headers":[{"level":2,"title":"前言","slug":"前言","link":"#前言","children":[]},{"level":2,"title":"1 .ipynb转.py文件流程","slug":"_1-ipynb转-py文件流程","link":"#_1-ipynb转-py文件流程","children":[]}],"relativePath":"posts/2023-06-30-yolov7-keypoint.md"}'),p={name:"posts/2023-06-30-yolov7-keypoint.md"},o=l(`<h1 id="yolov7-对人体关节点进行检测" tabindex="-1">Yolov7 ：对人体关节点进行检测 <a class="header-anchor" href="#yolov7-对人体关节点进行检测" aria-hidden="true">#</a></h1><h2 id="前言" tabindex="-1">前言 <a class="header-anchor" href="#前言" aria-hidden="true">#</a></h2><p>使用Yolov7来检测人体关键点， <strong>tools</strong>文件下<strong>keypoint.ipynb</strong>即可对图片的人体关键点检测,需要官网下载<strong>yolov7-w6-pose.pt</strong>权重文件</p><h2 id="_1-ipynb转-py文件流程" tabindex="-1">1 .ipynb转.py文件流程 <a class="header-anchor" href="#_1-ipynb转-py文件流程" aria-hidden="true">#</a></h2><div class="language-bash"><button title="Copy Code" class="copy"></button><span class="lang">bash</span><pre class="shiki material-theme-palenight" tabindex="0"><code><span class="line"><span style="color:#FFCB6B;">pip</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">install</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">jupyter</span></span>
<span class="line"><span style="color:#FFCB6B;">jupyter</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">nbconvert</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">--to</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">python</span><span style="color:#A6ACCD;"> </span><span style="color:#C3E88D;">tools/keypoint.ipynb</span></span>
<span class="line"></span></code></pre></div><p>即生成keypoint.py文件，如下：</p><div class="language-python"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki material-theme-palenight" tabindex="0"><code><span class="line"><span style="color:#676E95;font-style:italic;"># coding: utf-8</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> torch</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> cv2</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">from</span><span style="color:#A6ACCD;"> torchvision </span><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> transforms</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> numpy </span><span style="color:#89DDFF;font-style:italic;">as</span><span style="color:#A6ACCD;"> np</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">from</span><span style="color:#A6ACCD;"> utils</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">datasets </span><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> letterbox</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">from</span><span style="color:#A6ACCD;"> utils</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">general </span><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> non_max_suppression_kpt</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">from</span><span style="color:#A6ACCD;"> utils</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">plots </span><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> output_to_keypoint</span><span style="color:#89DDFF;">,</span><span style="color:#A6ACCD;"> plot_skeleton_kpts</span></span>
<span class="line"></span>
<span class="line"><span style="color:#A6ACCD;">device </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">device</span><span style="color:#89DDFF;">(</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">cuda:0</span><span style="color:#89DDFF;">&quot;</span><span style="color:#82AAFF;"> </span><span style="color:#89DDFF;font-style:italic;">if</span><span style="color:#82AAFF;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">cuda</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">is_available</span><span style="color:#89DDFF;">()</span><span style="color:#82AAFF;"> </span><span style="color:#89DDFF;font-style:italic;">else</span><span style="color:#82AAFF;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">cpu</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">weigths </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">load</span><span style="color:#89DDFF;">(</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">yolov7-w6-pose.pt</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">map_location</span><span style="color:#89DDFF;">=</span><span style="color:#82AAFF;">device</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">model </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> weigths</span><span style="color:#89DDFF;">[</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">model</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">]</span></span>
<span class="line"><span style="color:#A6ACCD;">_ </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> model</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">float</span><span style="color:#89DDFF;">().</span><span style="color:#82AAFF;">eval</span><span style="color:#89DDFF;">()</span></span>
<span class="line"></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">if</span><span style="color:#A6ACCD;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">cuda</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">is_available</span><span style="color:#89DDFF;">():</span></span>
<span class="line"><span style="color:#A6ACCD;">    model</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">half</span><span style="color:#89DDFF;">().</span><span style="color:#82AAFF;">to</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">device</span><span style="color:#89DDFF;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#A6ACCD;">image </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> cv2</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">imread</span><span style="color:#89DDFF;">(</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">./person.jpg</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">image </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">letterbox</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">image</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">960</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">stride</span><span style="color:#89DDFF;">=</span><span style="color:#F78C6C;">64</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">auto</span><span style="color:#89DDFF;">=True)[</span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">]</span></span>
<span class="line"><span style="color:#A6ACCD;">image_ </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> image</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">copy</span><span style="color:#89DDFF;">()</span></span>
<span class="line"><span style="color:#A6ACCD;">image </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> transforms</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">ToTensor</span><span style="color:#89DDFF;">()(</span><span style="color:#82AAFF;">image</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">image </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">tensor</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">np</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">array</span><span style="color:#89DDFF;">([</span><span style="color:#82AAFF;">image</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">numpy</span><span style="color:#89DDFF;">()]))</span></span>
<span class="line"></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">if</span><span style="color:#A6ACCD;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">cuda</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">is_available</span><span style="color:#89DDFF;">():</span></span>
<span class="line"><span style="color:#A6ACCD;">    image </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> image</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">half</span><span style="color:#89DDFF;">().</span><span style="color:#82AAFF;">to</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">device</span><span style="color:#89DDFF;">)</span><span style="color:#A6ACCD;">   </span></span>
<span class="line"><span style="color:#A6ACCD;">output</span><span style="color:#89DDFF;">,</span><span style="color:#A6ACCD;"> _ </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">model</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">image</span><span style="color:#89DDFF;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#A6ACCD;">output </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">non_max_suppression_kpt</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">output</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">0.25</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">0.65</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">nc</span><span style="color:#89DDFF;">=</span><span style="color:#82AAFF;">model</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">yaml</span><span style="color:#89DDFF;">[</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">nc</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">],</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">nkpt</span><span style="color:#89DDFF;">=</span><span style="color:#82AAFF;">model</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">yaml</span><span style="color:#89DDFF;">[</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">nkpt</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">],</span><span style="color:#82AAFF;"> </span><span style="color:#A6ACCD;font-style:italic;">kpt_label</span><span style="color:#89DDFF;">=True)</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">with</span><span style="color:#A6ACCD;"> torch</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">no_grad</span><span style="color:#89DDFF;">():</span></span>
<span class="line"><span style="color:#A6ACCD;">    output </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">output_to_keypoint</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">output</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">nimg </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> image</span><span style="color:#89DDFF;">[</span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">].</span><span style="color:#82AAFF;">permute</span><span style="color:#89DDFF;">(</span><span style="color:#F78C6C;">1</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">2</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">)</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">*</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">255</span></span>
<span class="line"><span style="color:#A6ACCD;">nimg </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> nimg</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">cpu</span><span style="color:#89DDFF;">().</span><span style="color:#82AAFF;">numpy</span><span style="color:#89DDFF;">().</span><span style="color:#82AAFF;">astype</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">np</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">uint8</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#A6ACCD;">nimg </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> cv2</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">cvtColor</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">nimg</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> cv2</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">COLOR_RGB2BGR</span><span style="color:#89DDFF;">)</span></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">for</span><span style="color:#A6ACCD;"> idx </span><span style="color:#89DDFF;font-style:italic;">in</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">range</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">output</span><span style="color:#89DDFF;">.</span><span style="color:#F07178;">shape</span><span style="color:#89DDFF;">[</span><span style="color:#F78C6C;">0</span><span style="color:#89DDFF;">]):</span></span>
<span class="line"><span style="color:#A6ACCD;">    </span><span style="color:#82AAFF;">plot_skeleton_kpts</span><span style="color:#89DDFF;">(</span><span style="color:#82AAFF;">nimg</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> output</span><span style="color:#89DDFF;">[</span><span style="color:#82AAFF;">idx</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">7</span><span style="color:#89DDFF;">:].</span><span style="color:#F07178;">T</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> </span><span style="color:#F78C6C;">3</span><span style="color:#89DDFF;">)</span></span>
<span class="line"></span>
<span class="line"><span style="color:#676E95;font-style:italic;"># save result</span></span>
<span class="line"><span style="color:#A6ACCD;">cv2</span><span style="color:#89DDFF;">.</span><span style="color:#82AAFF;">imwrite</span><span style="color:#89DDFF;">(</span><span style="color:#89DDFF;">&#39;</span><span style="color:#C3E88D;">person_keypoint.jpg</span><span style="color:#89DDFF;">&#39;</span><span style="color:#89DDFF;">,</span><span style="color:#82AAFF;"> nimg</span><span style="color:#89DDFF;">)</span></span>
<span class="line"></span></code></pre></div><p>如果需要对视频进行检测，则如下keypointvideo.py文件，代码如下：</p><div class="language-"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki material-theme-palenight" tabindex="0"><code><span class="line"><span style="color:#A6ACCD;">import torch</span></span>
<span class="line"><span style="color:#A6ACCD;">import cv2</span></span>
<span class="line"><span style="color:#A6ACCD;">import numpy as np</span></span>
<span class="line"><span style="color:#A6ACCD;">import time</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">from torchvision import transforms</span></span>
<span class="line"><span style="color:#A6ACCD;">from utils.datasets import letterbox</span></span>
<span class="line"><span style="color:#A6ACCD;">from utils.general import non_max_suppression_kpt</span></span>
<span class="line"><span style="color:#A6ACCD;">from utils.plots import output_to_keypoint, plot_skeleton_kpts</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">device = torch.device(&quot;cuda:0&quot; if torch.cuda.is_available() else &quot;cpu&quot;)</span></span>
<span class="line"><span style="color:#A6ACCD;">weigths = torch.load(&#39;yolov7-w6-pose.pt&#39;)</span></span>
<span class="line"><span style="color:#A6ACCD;">model = weigths[&#39;model&#39;]</span></span>
<span class="line"><span style="color:#A6ACCD;">model = model.half().to(device)</span></span>
<span class="line"><span style="color:#A6ACCD;">_ = model.eval()</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">cap = cv2.VideoCapture(&#39;kunkun.mp4&#39;)</span></span>
<span class="line"><span style="color:#A6ACCD;">if (cap.isOpened() == False):</span></span>
<span class="line"><span style="color:#A6ACCD;">    print(&#39;open failed.&#39;)</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;"># fps</span></span>
<span class="line"><span style="color:#A6ACCD;">frame_width = int(cap.get(3))</span></span>
<span class="line"><span style="color:#A6ACCD;">frame_height = int(cap.get(4))</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]</span></span>
<span class="line"><span style="color:#A6ACCD;">resize_height, resize_width = vid_write_image.shape[:2]</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;"># save video</span></span>
<span class="line"><span style="color:#A6ACCD;">out = cv2.VideoWriter(&quot;result_keypoint.mp4&quot;, </span></span>
<span class="line"><span style="color:#A6ACCD;">                      cv2.VideoWriter_fourcc(*&#39;mp4v&#39;), 30, </span></span>
<span class="line"><span style="color:#A6ACCD;">                      (resize_width, resize_height))</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">frame_count = 0</span></span>
<span class="line"><span style="color:#A6ACCD;">total_fps = 0</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">while(cap.isOpened):</span></span>
<span class="line"><span style="color:#A6ACCD;">    ret, frame = cap.read()</span></span>
<span class="line"><span style="color:#A6ACCD;">    if ret:</span></span>
<span class="line"><span style="color:#A6ACCD;">        orig_image = frame</span></span>
<span class="line"><span style="color:#A6ACCD;">        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)</span></span>
<span class="line"><span style="color:#A6ACCD;">        image = letterbox(image, (frame_width), stride=64, auto=True)[0]</span></span>
<span class="line"><span style="color:#A6ACCD;">        image_ = image.copy()</span></span>
<span class="line"><span style="color:#A6ACCD;">        image = transforms.ToTensor()(image)</span></span>
<span class="line"><span style="color:#A6ACCD;">        image = torch.tensor(np.array([image.numpy()]))</span></span>
<span class="line"><span style="color:#A6ACCD;">        image = image.to(device)</span></span>
<span class="line"><span style="color:#A6ACCD;">        image = image.half()</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">        start_time = time.time()</span></span>
<span class="line"><span style="color:#A6ACCD;">        with torch.no_grad():</span></span>
<span class="line"><span style="color:#A6ACCD;">            output, _ = model(image)</span></span>
<span class="line"><span style="color:#A6ACCD;">        end_time = time.time()</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">        # count fps</span></span>
<span class="line"><span style="color:#A6ACCD;">        fps = 1 / (end_time - start_time)</span></span>
<span class="line"><span style="color:#A6ACCD;">        total_fps += fps</span></span>
<span class="line"><span style="color:#A6ACCD;">        frame_count += 1</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml[&#39;nc&#39;], nkpt=model.yaml[&#39;nkpt&#39;], kpt_label=True)</span></span>
<span class="line"><span style="color:#A6ACCD;">        output = output_to_keypoint(output)</span></span>
<span class="line"><span style="color:#A6ACCD;">        nimg = image[0].permute(1, 2, 0) * 255</span></span>
<span class="line"><span style="color:#A6ACCD;">        nimg = nimg.cpu().numpy().astype(np.uint8)</span></span>
<span class="line"><span style="color:#A6ACCD;">        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)</span></span>
<span class="line"><span style="color:#A6ACCD;">        for idx in range(output.shape[0]):</span></span>
<span class="line"><span style="color:#A6ACCD;">            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">        # show fps</span></span>
<span class="line"><span style="color:#A6ACCD;">        cv2.putText(nimg, f&quot;{fps:.3f} FPS&quot;, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,</span></span>
<span class="line"><span style="color:#A6ACCD;">                    1, (0, 255, 0), 2)</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">        # 显示结果并保存</span></span>
<span class="line"><span style="color:#A6ACCD;">        cv2.imshow(&#39;image&#39;, nimg)</span></span>
<span class="line"><span style="color:#A6ACCD;">        out.write(nimg)</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">        # 按q退出</span></span>
<span class="line"><span style="color:#A6ACCD;">        if cv2.waitKey(1) &amp; 0xFF == ord(&#39;q&#39;):</span></span>
<span class="line"><span style="color:#A6ACCD;">            break</span></span>
<span class="line"><span style="color:#A6ACCD;">    else:</span></span>
<span class="line"><span style="color:#A6ACCD;">        break</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;">cap.release()</span></span>
<span class="line"><span style="color:#A6ACCD;">cv2.destroyAllWindows()</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span>
<span class="line"><span style="color:#A6ACCD;"># count avg fps</span></span>
<span class="line"><span style="color:#A6ACCD;">avg_fps = total_fps / frame_count</span></span>
<span class="line"><span style="color:#A6ACCD;">print(f&quot;Average FPS: {avg_fps:.3f}&quot;)</span></span>
<span class="line"><span style="color:#A6ACCD;"></span></span></code></pre></div>`,9),e=[o];function t(c,r,y,D,F,A){return n(),a("div",null,e)}const m=s(p,[["render",t]]);export{C as __pageData,m as default};
