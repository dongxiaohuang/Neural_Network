import matplotlib.pyplot as plt
##dropout
fig, ax = plt.subplots()
drop_out = [0.1, 0.2, 0.3, 0.5 ]
train_acc = [0.95, .85, .73, 0.47 ]# max acc
val_acc = [0.46, 0.45, .42, .40 ]

train_plt = ax.plot(drop_out, train_acc, '-o', label = 'max train accuracy')
val_plt = ax.plot(drop_out, val_acc, '-o', label = 'max test accuracy')
plt.ylabel('Performance')
plt.xlabel("Dropout Probability")
plt.legend()
vals_y = ax.get_yticks()
vals_x = ax.get_xticks()
ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals_x])
ax.set_yticklabels(['{:3.0f}%'.format(y*100) for y in vals_y])
# plt.show()
plt.savefig('dropout_compare.png')

# ## L2 compare
# l2 = [0, 0.1, 0.2, 0.3, 0.5 ]
# loss = [2.3, 1.1, 1.6 ,1.8,2 ]# max acc
# val_acc = [0.43, 0.5, 0.43, 0.45, 0.43]# max acc
# train_acc = [0.52, 0.9, 0.75, 0.65, 0.47]# max acc
#
# plt.subplot(1,2,1)
# plt.ylabel("Training loss")
# plt.xlabel("L2 regularisation rate")
# loss_plt = plt.plot(l2, loss, '-o', label = 'loss after 8000 iteration')
# plt.legend()
#
# plt.subplot(1,2,2)
#
# train_plt = plt.plot(l2, train_acc, '-o', label = 'max train accuracy')
# val_plt = plt.plot(l2, val_acc, '-o', label = 'max test accuracy')
# plt.ylabel('Performance')
# plt.xlabel("L2 regularisation rate")
# plt.legend()
# plt.savefig('l2_compare.png')
# plt.show()


################# Learning Rate ##########################

# plt.subplot(1,2,1)
# lr = [1, 0.01, 0.001, 0.0001, 0.00001, 0.005,.0005 ]
# loss = [0.15, 1, 0.1 , .8, 1.5,.75, 0.07 ]# max acc
# lr, tl = zip(*sorted(zip(lr, loss)))
# plt.legend(loc=0)
# plt.ylabel("Training loss")
# plt.xlabel("Learning Rate")
# plt.xscale('log')
# loss_plt = plt.plot(lr, tl, '-or', label = 'loss after 9000 iteration')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# lr = [1, 0.01, 0.001, 0.0001, 0.00001, 0.005,.0005 ]
# val_acc = [0.45, 0.35, 0.42,.41,.35, 0.35, 0.43]
# train_acc = [1, 0.53, 1, .72, 0.38, 0.68, 1 ]
# loss = [0.15, 1, 0.1 , .8, 1.5,.75, 0.07 ]# max acc
# lr, va = zip(*sorted(zip(lr, val_acc)))
# lr, ta = zip(*sorted(zip(lr, train_acc)))
# plt.plot(lr, va, '-o', label='validation')
# plt.plot(lr, ta, '-o', label='training')
# plt.legend(loc=0)
# # lr, tl = zip(*sorted(zip(lr, loss)))
# # ax2 = ax1.twinx()
# # ax2.plot(lr[:40], tl[:40], 'r-o', label='training loss')
# # ax2.legend(loc=0)
# plt.xlabel('Learning Rate')
# plt.ylabel('Accuracy')
# plt.xscale('log')
# plt.tight_layout()
# plt.savefig('lr_compare.png')
# plt.show()
