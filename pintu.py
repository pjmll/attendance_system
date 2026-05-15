import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps

def create_image_grid(image_paths, output_path, cols=None, target_size=(400, 400)):
    """
    将多张图片拼接成等大小的宫格。
    """
    num_images = len(image_paths)
    if num_images == 0:
        print("错误：没有提供任何图片路径！")
        return

    # 1. 计算宫格的列数和行数
    if cols is None:
        cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # 2. 创建一张空白的背景画布 (白色背景)
    grid_width = cols * target_size[0]
    grid_height = rows * target_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # 3. 遍历图片，调整大小并贴入画布
    for index, img_path in enumerate(image_paths):
        try:
            with Image.open(img_path) as img:
                # 使用 ImageOps.fit 居中裁剪并缩放，保证图片不被拉伸变形
                img_resized = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

                # 计算当前图片在画布上的 (x, y) 坐标
                row = index // cols
                col = index % cols
                x = col * target_size[0]
                y = row * target_size[1]

                # 将处理好的小图粘贴到大画布的对应位置
                grid_image.paste(img_resized, (x, y))
                
        except Exception as e:
            print(f"跳过图片 {img_path}，读取或处理时发生错误: {e}")

    # 4. 保存最终图片
    grid_image.save(output_path, quality=95)
    print(f"🎉 拼接成功！生成的宫格图已保存至: {output_path}")


# ================= 交互式运行逻辑 =================
if __name__ == "__main__":
    # 初始化 tkinter，并隐藏无用的小主窗口
    root = tk.Tk()
    root.withdraw()

    print("请在弹出的窗口中选择你要拼接的图片...")
    
    # 弹出文件选择对话框（支持多选）
    selected_files = filedialog.askopenfilenames(
        title="1. 请选择要拼接的图片（可按住 Ctrl 或 Shift 多选）",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.webp *.bmp")]
    )

    # 检查用户是否真的选择了图片
    if not selected_files:
        print("❌ 你没有选择任何图片，程序已退出。")
    else:
        print(f"✅ 你一共选择了 {len(selected_files)} 张图片。")
        
        print("请在弹出的窗口中选择你要保存的位置和文件名...")
        # 弹出保存文件对话框
        save_path = filedialog.asksaveasfilename(
            title="2. 请选择拼接后图片的保存位置",
            defaultextension=".jpg", # 默认保存为 jpg 格式
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png")],
            initialfile="我的宫格拼图.jpg" # 默认文件名
        )

        if not save_path:
            print("❌ 你取消了保存，程序已退出。")
        else:
            print("⏳ 正在拼命处理中，请稍候...")
            
            # 执行拼接代码
            # 如果你想自定义列数，可以将 cols 修改为你想要的数字，比如 cols=3
            # 如果 cols=None，程序会自动计算成尽量接近正方形的排版
            create_image_grid(
                image_paths=selected_files, 
                output_path=save_path, 
                cols=None, 
                target_size=(500, 500)
            )