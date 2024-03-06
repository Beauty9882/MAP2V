import os
import json
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchvision import utils, transforms
from torch.utils.tensorboard import SummaryWriter

from encoder.fetch import fetch_encoder
from encoder.blackbox_encoder import BlackboxEncoder,BlackboxEncoder4Adaface
from encoder.blackbox_encoder import WhiteboxEncoder,WhiteboxEncoder4Adaface 
from generator.stylegan_utils import StyleGANWrapper
from dataset.parse_dataset import dataset_parser
from utils import cosine_similarity, str2bool


 
def ZOGE(args, latent, G, v_T, T):
    '''
    Zeroth-Order Gradient Estimation
    latent: current latent vector
    G: StyleGAN generator
    v_T: target feature vector (ground-truth)
    T: target encoder
    return: estimated gradient (same shape as latent)
    '''
    mu = args.mu
    ndirs = args.ndirs
    max_batch = args.batch_size
    d = latent.size(-1)
    grad_est = torch.zeros_like(latent)

    with torch.no_grad():
        x_G = G(latent, args.init_latent)
        v_G = T(x_G, flip=True)
        loss = 1 - cosine_similarity(v_G, v_T).mean()

        for i in range(0, ndirs, max_batch):
            B = min(max_batch, ndirs - i)
            if args.latent_space == 'Wp' or args.latent_space == 'Wp_FFHQ':
                l_rep = latent.repeat(B, 1, 1)
                u = torch.randn_like(l_rep)
                u = u / u.norm(dim=[1,2],keepdim=True)            
            else:
                l_rep = latent.repeat(B, 1)
                u = torch.randn_like(l_rep)
                u = F.normalize(u, dim=1)  # sample from unit sphere
            l_mod = l_rep + (mu * u)
            x_mod = G(l_mod, args.init_latent)
            v_mod = T(x_mod, flip=True)
            loss_mod = 1 - cosine_similarity(v_mod, v_T)
            if args.latent_space == 'Wp' or args.latent_space == 'Wp_FFHQ':
                grad_temp = (d * (loss_mod - loss) / mu).view(-1, 1, 1) * u
            else:
                grad_temp = (d * (loss_mod - loss) / mu).view(-1, 1) * u
            grad_est += grad_temp.sum(dim=0, keepdim=True)
    grad_est /= ndirs

    return grad_est, loss, x_G, v_G



def generate_random_init_samples(args, num_samples=512):
    # check if random samples already exists
    project_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(f'{project_dir}/random_init_samples.pth'):
        # fetch generator
        stgan = StyleGANWrapper(args)

        # sample random Z-latents
        z = torch.randn(num_samples, 512).to(args.device)
        with torch.no_grad():
            w = stgan.generator.style(z)
        z, w = z.cpu(), w.cpu()

        # save
        torch.save({'latents_z':z, 'latents_w':w},
                   f'{project_dir}/random_init_samples.pth')



def main(args):
    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


    # iteration for visualization
    N = args.num_iters
    log_iter = [int(N * 0.1), int(N * 0.2),  int(N * 0.3),int(N * 0.4), int(N * 0.5), int(N * 0.6),  int(N * 0.7), int(N * 0.8),int(N * 0.9), N]  # logging at 10%, 20%, 40%, 100%
    writer = SummaryWriter(log_dir=args.save_dir)


    # encoder dict: contains all necessary stuff w.r.t. each encoders
    enc_dict = {name:{} for name in args.encoder_list}
    for name in args.encoder_list:
        # fetch encoders
        enc = fetch_encoder(name, device = args.device)
        if name == args.enc_tgt:
            args.align = enc.align  # face align scheme of target encoder
        if name == args.enc_tgt and not args.blackbox:
             if args.enc_tgt =='AdaFace':
                enc = WhiteboxEncoder4Adaface(enc, img_size=enc.img_size).to(args.device)
            #  elif args.enc_tgt =='Due':
            #      enc = WhiteboxEncoder4Due(enc, img_size=enc.img_size).to(args.device)
             else:
                enc = WhiteboxEncoder(enc, img_size=enc.img_size).to(args.device)
        else:
            if args.enc_tgt =='AdaFace':
                enc = BlackboxEncoder4Adaface(enc, img_size=enc.img_size).to(args.device)
            else:
                enc = BlackboxEncoder(enc, img_size=enc.img_size).to(args.device)
        enc_dict[name]['enc'] = enc
        # cosine log
        enc_dict[name]['cosine'] = np.zeros(shape=(0, args.num_iters+1))


    # fetch generator
    stgan = StyleGANWrapper(args)


    # load pre-generated random samples
    project_dir = os.path.abspath(os.path.dirname(__file__))
    
    if args.init_type == 'FFHQ':
        init_samples = torch.load('/media/Storage2/zh/face-privacy/MAP2V/latents/invert_w_256.pt', map_location='cpu')#70000 14 512
    elif args.init_type == 'random':
        init_samples = torch.load(f'{project_dir}/random_init_samples.pth', map_location='cpu')
    elif args.init_type == 'cluster':
        init_samples = torch.load('/media/Storage1/dong/random_init_cluster_samples_AdaFace_40000_512.pth', map_location='cpu')
    
    if args.latent_space == 'Z':
        init_latents = init_samples['latents_z']
    elif args.latent_space == 'Wp_FFHQ':
        np.random.seed(0)
        init_latents_all = init_samples
        # random choose num_inits from init_latents
        random_indices = np.random.choice(len(init_latents_all), size=args.num_inits, replace=False)
        init_latents = init_latents_all[random_indices]
    else:
        init_latents = init_samples['latents_w']

    init_latents = torch.FloatTensor(init_latents).to(args.device)
    init_feats = torch.FloatTensor().to(args.device)
    for beg in tqdm(range(0, args.num_inits, args.batch_size)): # attention, we only list from 0 to num_inits 
        end = min(args.num_inits, beg + args.batch_size)
        latents = init_latents[beg:end] #ffhq,(batch_size,14,512)
        with torch.no_grad():
            init_image = stgan(latents) #(160,3,192,192)
            feat = enc_dict[args.enc_tgt]['enc'](init_image, flip=True)
            init_feats = torch.cat((init_feats, feat), dim=0)


    # fetch target face images
    targets, imgdirs = dataset_parser(args)#targets:list len=200 all 'name.jpg'
    num_targets = len(targets)

    # standard image transform
    resize = transforms.Resize((args.crop_size, args.crop_size))
    if args.enc_tgt=='MagFace':
        trf =trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],std=[1., 1., 1.]),
        ])
    else:
        trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])


    # reconstruct
    for cnt in range(num_targets):
        target = targets[cnt]
        imgdir = imgdirs[cnt]
        print(f"{cnt}: {target}, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # log cosine similarity w.r.t. all encoders
        for name in args.encoder_list:
            enc_dict[name]['cosine'] = np.concatenate((enc_dict[name]['cosine'], 
                                                       np.zeros((1, args.num_iters + 1))))
        
        # align image & compute target feature
        with torch.no_grad():
            img = trf(Image.open(imgdir))#3 112 112
            img = img.unsqueeze(0).to(args.device)#1 3 112 112

            # compute target features
            for name in args.encoder_list:
                feat = enc_dict[name]['enc'](img, flip=True)
                enc_dict[name]['target'] = feat.clone()

            # initialize latent
            if args.fixed_init:
                args.init_latent = stgan.generator.mean_latent(int(1e4)).clone()
                # stgan.generator.mean_latent(int(1e4))*0.8
            else:  # ensemble init.
                cosine = cosine_similarity(enc_dict[args.enc_tgt]['target'], init_feats).squeeze()
                _, idx = cosine.topk(args.topk)
                args.init_latent = init_latents[idx].mean(dim=0, keepdim=True)  # latent ensembling

            init_image = stgan(args.init_latent)#1 3 192 192
            for name in args.encoder_list:
                feat = enc_dict[name]['enc'](init_image, flip=True)
                init_cos = cosine_similarity(enc_dict[name]['target'], feat)
                enc_dict[name]['cosine'][cnt, 0] = init_cos.item()
                print("{}:{:.3f}".format(name, init_cos.item()), end=", ")
            print()

        latent = args.init_latent.clone() #1 14 512
        if args.latent_space == 'Wp' and args.init_latent.shape[1]!=14:
            latent = latent.unsqueeze(1).repeat(1, stgan.generator.n_latent, 1)

        latent.requires_grad = True
        optimizer = optim.Adam([latent], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        writer.add_scalar("lr", args.lr, 0)

        query_count = 0
        img_seq = init_image.cpu()
        for i in tqdm(range(1, args.num_iters+1)):
            if args.blackbox:
                query_count += (args.ndirs + 1)
                grad, loss, x_G, v_G = ZOGE(args, latent, stgan, 
                                            enc_dict[args.enc_tgt]['target'], 
                                            enc_dict[args.enc_tgt]['enc'])
                optimizer.zero_grad()
                latent.backward(grad)
            else:
                x_G = stgan(latent, args.init_latent)
                v_G = enc_dict[args.enc_tgt]['enc'](x_G, flip=True)
                loss = 1 - cosine_similarity(v_G, enc_dict[args.enc_tgt]['target']).mean()
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()
            scheduler.step()

            # tensorboard logging
            if i % args.cosine_log_freq == 0:
                for name in args.encoder_list:
                    if name == args.enc_tgt:
                        enc_dict[name]['cosine'][cnt, i] = 1 - loss.item()
                    else:
                        with torch.no_grad():
                            v_G = enc_dict[name]['enc'](x_G, flip=True)
                            cos = cosine_similarity(v_G, enc_dict[name]['target'])
                            enc_dict[name]['cosine'][cnt, i] = cos.item()
                    # print(enc_dict[name]['cosine'][cnt, i])

            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("lr", lr, i + 1)

            if i in log_iter:  # visualization at 10%, 20%, 40%, 100%
                with torch.no_grad():
                    img_gen, ltn_gen = stgan(latent, args.init_latent, return_latents=True)
                img_gen = resize(img_gen).cpu()
                img_seq = torch.cat((img_seq, img_gen), dim=0)
                for name in args.encoder_list:
                    print("{}:{:.3f},".format(name, enc_dict[name]['cosine'][cnt, i]), end=", ")
                print()

        # update tensorboard
        for name in args.encoder_list:
            for i in tqdm(range(0, args.num_iters + 1)):
                writer.add_scalar(name, enc_dict[name]['cosine'][:, i].mean(), i)
            writer.flush()

        # save image & image sequences  lfw-200、celeba
        img = resize(img).cpu()
        img_seq = torch.cat((img_seq, img))
        if args.dataset == 'celeba':
            utils.save_image(img_gen, f'{args.save_dir}/attack_images/{target[:-11]}_{target[-10:]}', nrow=1, normalize=True, range=(-1, 1))
            utils.save_image(img_seq, f'{args.save_dir}/image_sequence/{target[:-11]}_{target[-10:]}',
                            nrow=img_seq.size(0), normalize=True, range=(-1, 1))
            ltn_gen = ltn_gen.detach().cpu().numpy()
            np.save(f'{args.save_dir}/attack_latents/{target[:-11]}_{target[-10:]}', ltn_gen)
        else:
            utils.save_image(img_gen, f'{args.save_dir}/attack_images/{target}', nrow=1, normalize=True, range=(-1, 1))
            utils.save_image(img_seq, f'{args.save_dir}/image_sequence/{target}',
                            nrow=img_seq.size(0), normalize=True, range=(-1, 1))
            # save latent
            ltn_gen = ltn_gen.detach().cpu().numpy()
            np.save(f'{args.save_dir}/attack_latents/{target[:-4]}', ltn_gen)
        

    # close tensorboard logger
    writer.close()

    # save cosine log
    for name in args.encoder_list:
        np.save(f"{args.save_dir}/{name}_cosine_log", enc_dict[name]['cosine'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # optimization parameters
    parser.add_argument('--num_iters', default=400, type=int, help='number of iteration per reconstruction')
    parser.add_argument('--ndirs', default=16, type=int, help='number of perturbation directions for ZOGE')
    parser.add_argument('--mu', default=0.1, type=float, help='small constant for ZOGE')
    parser.add_argument('--latent_space', default='Wp_FFHQ', type=str, help='which latent space to use: [Z, W, Wp,Wp_FFHQ], Wp_FFHQ for inversion latent')
    parser.add_argument('--init_type', default='FFHQ', type=str, help='which latent to use: [FFHQ,random,cluster], Wp_FFHQ for inversion latent. when FFHQ, latentspace should be Wp_FFHQ. Other latentspace is Z Wp ')

    # optimization parameters 2
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate; cosine decay is used')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--milestones', default=[100], type=list)
    parser.add_argument('--gamma', default=0.1, type=float)

    # initialization parameters
    parser.add_argument('--fixed_init', default=False, type=str2bool, help='use fixed initialization') # must be false, man!
    parser.add_argument('--num_inits', default=500, type=int, help='number of init latents')
    parser.add_argument('--topk', default=5, type=int, help='topk for latent ensembling')

    # target dataset & encoder
    parser.add_argument('--dataset', default='lfw-200', type=str, help='target dataset to attack') #we have:lfw-200、celeba cfp-fp-200-F
    parser.add_argument('--enc_tgt', default='Due', type=str, help='target encoder type')
    parser.add_argument('--blackbox', default=True, type=str2bool, help='set target encoder as a blackbox')
    parser.add_argument('--encoder_list', default=['Due'], type=list,
                        help='encoders for measuring cosine similarity during optimization') #Due PartialFace DCTDP  ArcFace MagFace AdaFace 

    # StyleGAN & alignment model parameters - need not change
    parser.add_argument('--resolution', default=256, type=int, help='StyleGAN output resolution')
    parser.add_argument('--batch_size', default=170, type=int, help='StyleGAN batch size. Reduce to avoid OOM')
    parser.add_argument('--truncation', default=0.8, type=int, help='interpolation weight w.r.t. initial latent')
    parser.add_argument('--generator_type', default='FFHQ-256', type=str)
    parser.add_argument('--crop_size', default=192, type=int, help='crop size for StyleGAN output')

    # Misc.
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--device_id', default=0, type=int, help='which gpu to use')
    parser.add_argument('--cosine_log_freq', default=10, type=int, help='how often to log cosine similarity')

    args = parser.parse_args()
    args.device = f'cuda:{args.device_id}'


    # check required arguments
    if args.latent_space not in ['Z', 'W', 'Wp','Wp_FFHQ']:
        raise NotImplementedError(f"Invalid argument:{args.latent_space}. "
                                  f"--latent_space must be one of ['Z', 'W', 'Wp','Wp_FFHQ'].")

    # make directory for saving results
    box = 'BB' if args.blackbox else 'WB'
    args.query_per_attack = args.num_iters * (args.ndirs + 1)
    args.save_dir = f'/media/Storage2/zh/face-privacy/MAP2V/results/{args.dataset}/{args.enc_tgt}/{args.seed}_{args.init_type}_{args.latent_space}_{args.num_inits}_{args.num_iters}_{args.lr}_{args.ndirs}_top_{args.topk}'
    
    
    os.makedirs(f'{args.save_dir}/attack_images', exist_ok=True)
    os.makedirs(f'{args.save_dir}/attack_latents', exist_ok=True)
    os.makedirs(f'{args.save_dir}/image_sequence', exist_ok=True)

    # save arguments
    argdict = args.__dict__.copy()
    with open(f'{args.save_dir}/args.txt', 'w') as fp:
        json.dump(argdict, fp, indent=2)

    # generate random samples for initialization
    generate_random_init_samples(args)

    # reconstruct
    main(args)
