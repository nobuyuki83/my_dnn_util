import torch
import numpy, cv2

def write_grid_img(net_gen1,batches,path):
#  net_gen1.eval()
  wh = 224
  np_out0 = numpy.ndarray((0,wh,wh,3),dtype=numpy.uint8)
  for itr in range(8):
    vpt_out_real, vpt_out_fake, vpt_tg_fake = batches.get_batch_vpt_realfake(net_gen1, requires_grad_fake=False)
    np_img = numpy.moveaxis(vpt_out_fake[:,:3,:,:].cpu().data.numpy(), 1, 3)
    np_batch_bgr1 =( (np_img+1.0) * (255.0*0.5) ).astype(numpy.uint8)
    np_out0 = numpy.vstack((np_out0,np_batch_bgr1))
    if np_out0.shape[0] > 8:
       break
  np_img = numpy.ndarray((wh*2,wh*4,3),dtype=numpy.uint8)
  np_img[0 * wh:1 * wh, 0 * wh:1 * wh, :] = np_out0[0, :, :, :]
  np_img[1 * wh:2 * wh, 0 * wh:1 * wh, :] = np_out0[1, :, :, :]
  np_img[0 * wh:1 * wh, 1 * wh:2 * wh, :] = np_out0[2, :, :, :]
  np_img[1 * wh:2 * wh, 1 * wh:2 * wh, :] = np_out0[3, :, :, :]
  np_img[0 * wh:1 * wh, 2 * wh:3 * wh, :] = np_out0[4, :, :, :]
  np_img[1 * wh:2 * wh, 2 * wh:3 * wh, :] = np_out0[5, :, :, :]
  np_img[0 * wh:1 * wh, 3 * wh:4 * wh, :] = np_out0[6, :, :, :]
  np_img[1 * wh:2 * wh, 3 * wh:4 * wh, :] = np_out0[7, :, :, :]
  cv2.imwrite(path,np_img)


def optD(net_gen1,net_dis1,batches,criterion_DG,optimizer_D,itr):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net_gen1.eval()
  net_dis1.train()
  for params in net_gen1.parameters(): params.requires_grad = False
  for params in net_dis1.parameters(): params.requires_grad = True
  ####
  optimizer_D.zero_grad()
  vpt_bgr_real,vpt_bgr_fake,tg_fake = batches.get_batch_vpt_realfake(net_gen1, requires_grad_fake=False)
  vpt_bgr_real.requires_grad = True
  vpt_bgr_fake.requires_grad = True
  out_real = net_dis1(vpt_bgr_real)
  out_fake = net_dis1(vpt_bgr_fake)
  tg_real = torch.autograd.Variable(torch.ones_like(tg_fake),requires_grad=False).to(device)
  D_real_loss = criterion_DG(out_real, tg_real)
  D_fake_loss = criterion_DG(out_fake, tg_fake)
  print('{{"metric": "disc real loss", "value": {0}, "step":{1} }}'.format(D_real_loss, itr))
  print('{{"metric": "disc fake loss", "value": {0}, "step":{1} }}'.format(D_fake_loss, itr))
  (D_real_loss + D_fake_loss).backward()
  optimizer_D.step()


def optG_adv(net_gen1, net_dis1, batches, criterion_DG, optimizer_G, itr:int):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net_gen1.train()
  net_dis1.eval()
  for params in net_gen1.parameters(): params.requires_grad = True
  for params in net_dis1.parameters(): params.requires_grad = False
  ####
  optimizer_G.zero_grad()
  vpt_bgr_real, vpt_bgr_fake, tg_fake = batches.get_batch_vpt_realfake(net_gen1,
                                                                       requires_grad_fake=True)
  out_fake = net_dis1(vpt_bgr_fake)
  tg_real = torch.autograd.Variable(torch.ones_like(tg_fake),requires_grad=False).to(device)
  G_loss = criterion_DG(out_fake, tg_real)
  print('{{"metric": "gen losss", "value": {0}, "step":{1} }}'.format(G_loss.cpu().data.numpy(), itr))
  G_loss.backward()
  optimizer_G.step()

def optG_pcp(net_gen, net_vgg, batches, optimizer_P, itr):
  net_gen.train()
  net_vgg.eval()
  for param in net_vgg.parameters(): param.requires_grad = False
  for params in net_gen.parameters(): params.requires_grad = True
  optimizer_P.zero_grad()
  vpt_real, vpt_fake, vpt_rmask = batches.get_batch_vpt_realfake(net_gen,
                                                                 requires_grad_fake=True)
  with torch.no_grad():
    vpt_batch_real_vgg = net_vgg.prep256(vpt_real[:,:3,:,:])
    vpt_real_out = net_vgg.forward(vpt_batch_real_vgg)
  vpt_batch_fake_vgg = net_vgg.prep256(vpt_fake[:,:3,:,:])
  vpt_fake_out = net_vgg.forward(vpt_batch_fake_vgg)
  P_loss = net_vgg.mseloss(vpt_fake_out,vpt_real_out)
  print('{{"metric": "pcp loss0", "value": {0}, "step":{1} }}'.format(P_loss.cpu().data.numpy(), itr))
  P_loss.backward()
  optimizer_P.step()


def train_gan(net_gen1, net_dis1, net_vgg, batches_G, batches_D,
              step_size_D=1.0e-5, step_size_G=1.0e-5,
              nitr=1000):

  print("train dis gen together")
  print("step_size_D",step_size_D)
  print("step_size_G",step_size_G)
  print("nitr",nitr)

  net_gen1.train()
  net_dis1.train()
  net_vgg.eval()

  for param in net_vgg.parameters(): param.requires_grad = False

  optimizer_D = torch.optim.Adam(net_dis1.parameters(), lr=step_size_D, betas=(0.9,0.999))
  optimizer_G = torch.optim.Adam(net_gen1.parameters(), lr=step_size_G, betas=(0.9,0.999))
  optimizer_P = torch.optim.Adam(net_gen1.parameters(), lr=2.0e-5)

#  criterion_DG = torch.nn.BCELoss()
  criterion_DG = torch.nn.MSELoss()

  for itr in range(nitr):
    for itr_D in range(1):
      optD(net_gen1, net_dis1, batches_D, criterion_DG, optimizer_D, itr*1+itr_D)
    optG_adv(net_gen1, net_dis1, batches_G, criterion_DG, optimizer_G, itr)
    optG_pcp(net_gen1, net_vgg, batches_G, optimizer_P, itr)

    if itr % 40 == 39:
      print("save_model dis gen")
      torch.save(net_gen1.state_dict(), net_gen1.path_file+str(itr))
      torch.save(net_dis1.state_dict(), net_dis1.path_file+str(itr))

    if itr % 10 == 0:
      write_grid_img(net_gen1,batches_G,"out_"+str(itr)+".png")


def train_D_initial(net_gen1, net_dis1,
            batches,
            step_size=1.0e-4, nitr=1000):
  print("train_dis")
  print("step_size",step_size)
  net_gen1.eval()
  net_dis1.train()

  optimizer_D = torch.optim.Adam(net_dis1.parameters(), lr=step_size)

#  criterion = torch.nn.BCELoss()
  criterion_DG = torch.nn.MSELoss()

  for itr in range(nitr):
    optD(net_gen1, net_dis1, batches, criterion_DG, optimizer_D, itr)

    if itr % 40 == 39:
      print("save_model dis")
      torch.save(net_dis1.state_dict(), net_dis1.path_file)


def train_G_perceptual(net_gen1, net_vgg, batches, step_size=1.0e-4, nitr=1000):
  print("train0",step_size,nitr)
  print("npix: ", net_gen1.npix)
  print("nstride: ", net_gen1.nstride)

  optimizer_P = torch.optim.Adam(net_gen1.parameters(), lr=step_size)
  for itr in range(nitr):
    optG_pcp(net_gen1, net_vgg, batches, optimizer_P, itr)

    if itr % 100 == 99:
      torch.save(net_gen1.state_dict(), net_gen1.path_file)

    if itr % 10 == 0:
      write_grid_img(net_gen1,batches,"out_"+str(itr)+".png")
