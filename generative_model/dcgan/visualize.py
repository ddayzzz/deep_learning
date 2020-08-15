import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


result_dir = './result'
G_losses = np.load(result_dir + '/g_loss.npy')
D_losses = np.load(result_dir + '/d_loss.npy')
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 绘制变化的曲线
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
img_list = np.load(result_dir + '/generated_images.npy')
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save(result_dir + '/generated_images_animation.mp4', writer)