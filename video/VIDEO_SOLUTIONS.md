# 视频访问问题解决方案

## 问题分析
- 视频文件大小：49MB
- GitHub文件大小限制：通常建议<25MB
- 格式：MOV格式在某些情况下可能不被GitHub完全支持

## 解决方案

### 方案1：压缩视频（推荐）
使用ffmpeg压缩视频到更小的尺寸：

```bash
# 安装ffmpeg（如果还没有）
brew install ffmpeg

# 压缩视频到10MB以下
ffmpeg -i "Screen Recording 2025-10-21 at 16.31.11.mov" \
  -vcodec libx264 \
  -crf 28 \
  -preset medium \
  -vf "scale=1280:720" \
  -acodec aac \
  -b:a 128k \
  demo_compressed.mp4
```

### 方案2：创建缩略图视频
创建一个短视频预览：

```bash
# 提取前30秒作为预览
ffmpeg -i "Screen Recording 2025-10-21 at 16.31.11.mov" \
  -t 30 \
  -vcodec libx264 \
  -crf 23 \
  -preset fast \
  -vf "scale=1280:720" \
  demo_preview.mp4
```

### 方案3：使用外部视频托管
1. **YouTube**：上传到YouTube并嵌入链接
2. **Vimeo**：专业视频托管平台
3. **GitHub Releases**：作为发布附件上传

### 方案4：创建GIF动画
将关键部分转换为GIF：

```bash
# 创建GIF动画（前10秒）
ffmpeg -i "Screen Recording 2025-10-21 at 16.31.11.mov" \
  -t 10 \
  -vf "fps=10,scale=640:-1:flags=lanczos,palettegen" \
  palette.png

ffmpeg -i "Screen Recording 2025-10-21 at 16.31.11.mov" \
  -i palette.png \
  -t 10 \
  -filter_complex "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  demo_animation.gif
```

## 当前状态
- ✅ 视频文件已上传到GitHub
- ✅ README.md已配置下载链接
- ✅ HTML视频标签已添加
- ⚠️ 由于文件大小，可能需要压缩或使用替代方案

## 建议
1. 先尝试压缩视频到<10MB
2. 如果仍有问题，考虑使用外部托管
3. 可以创建多个版本：完整版、预览版、GIF动画版
