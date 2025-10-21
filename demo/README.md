# Demo Videos

**Note**: Demo videos are actually located in the `../video/` directory.

This directory was created as a placeholder for future demo videos, but the current demonstration video is stored in the main `video/` folder.

## Video Guidelines

### Supported Formats
- **MP4** (recommended)
- **MOV**
- **WebM**

### File Size Recommendations
- Keep videos under **25MB** for optimal GitHub performance
- For larger videos, consider:
  - Compressing the video
  - Using external hosting (YouTube, Vimeo)
  - Using GitHub Releases for large files

### Naming Convention
- `demo.mp4` - Main system demonstration
- `api-demo.mp4` - API functionality demo
- `ui-demo.mp4` - User interface walkthrough
- `deployment-demo.mp4` - Deployment process demo

## Upload Instructions

1. **Add your video file** to this directory
2. **Update the README.md** with the correct video path
3. **Commit and push** to GitHub
4. **GitHub will automatically display** the video in the README

## Example Usage in README.md

```markdown
## 🎥 Demo Video

![System Demo](./demo/demo.mp4)
```

Or for inline display:

```html
<video width="100%" controls>
  <source src="./demo/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```
