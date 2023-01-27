import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


def get_differential_filter():
    filter_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    filter_x = filter_y.T

    return filter_x, filter_y


def filter_image(im, filter):
    im_filtered = np.zeros((im.shape[0], im.shape[1]))
    im_padded = np.pad(im, ((1,1),(1,1)), 'constant')

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_filtered[i,j] = np.sum(filter * im_padded[i:i+3,j:j+3])

    return im_filtered


def find_match(img1, img2):
    x1 = []
    x2 = []

    sift = cv2.xfeatures2d.SIFT_create()
    img1_kps, img1_descriptors = sift.detectAndCompute(img1, None)
    img2_kps, img2_descriptors = sift.detectAndCompute(img2, None)

    NN = NearestNeighbors().fit(img2_descriptors)
    dist, idx = NN.kneighbors(img1_descriptors)

    for i in range(len(dist)):
        if dist[i, 0] / dist[i, 1] < 0.7 :
            x1.append(img1_kps[i].pt)
            x2.append(img2_kps[idx[i, 0]].pt)

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    best_transform = None
    best_inliers = 0
    for i in range(ransac_iter):
        random_sample = np.random.choice(len(x1), 3, replace = False)

        A = np.array([[x1[random_sample][0][0], x1[random_sample][0][1], 1, 0, 0, 0], [0, 0, 0, x1[random_sample][0][0], x1[random_sample][0][1], 1], [x1[random_sample][1][0], x1[random_sample][1][1], 1, 0, 0, 0], [0, 0, 0, x1[random_sample][1][0], x1[random_sample][1][1], 1], [x1[random_sample][2][0], x1[random_sample][2][1], 1, 0, 0, 0], [0, 0, 0, x1[random_sample][2][0], x1[random_sample][2][1], 1],])
        b = x2[random_sample].reshape(-1)

        try:
            transform = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        transform = np.array(list(transform) + [0, 0, 1])
        transform = transform.reshape((3, 3))
        if best_transform is None:
            best_transform = transform

        inliers = 0
        for j in range(x1.shape[0]):
            template_point = np.array(list(x1[j]) + [1])
            target_point = np.array(list(x2[j]) + [1])
            template_point_image = np.matmul(transform, template_point)
            distance = np.sqrt(np.sum((template_point_image - target_point) ** 2))
            if distance < ransac_thr:
                inliers += 1
        if inliers > best_inliers:
            best_transform = transform
            best_inliers = inliers

    A = best_transform

    return A


def warp_image(img, A, output_size):
    img_warped = np.zeros(output_size)

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            x1 = np.array(([j], [i], [1]))
            x2 = np.floor(A @ x1)
            if x2[0] < img.shape[0] and x2[1] < img.shape[1]:
                img_warped[i][j] = img[int(x2[1]), int(x2[0])]

    return img_warped


def align_image(template, target, A):
    W = A

    template = template/np.max(template)
    target = target/np.max(target)

    filter_x, filter_y = get_differential_filter()

    dIdu = filter_image(template, filter_x)
    dIdv = filter_image(template, filter_y)

    hessian = np.zeros((6,6)) 
    for u in range(template.shape[0]):
        for v in range(template.shape[1]):
            prod = np.array([dIdu[u,v],dIdv[u,v]]) @ np.array([[u,v,1,0,0,0],[0,0,0,u,v,1]])
            hessian += np.multiply(np.vstack(prod), prod)
    hessian_inverse = np.linalg.pinv(hessian)
    
    err_list = []
    for i in range(10): #change this range to limit iterations and help with runtime, I used 10 for my writeup submission
        I_warped = warp_image(target, W, template.shape)
        
        I_error = I_warped - template
        error = np.linalg.norm(I_error)
        err_list.append(error)
        if(error < 120):
            break
        
        H = np.zeros(6)
        for u in range(template.shape[0]):
            for v in range(template.shape[1]):
                H += np.array([dIdu[u,v],dIdv[u,v]]) @ np.array([[u,v,1,0,0,0],[0,0,0,u,v,1]]) * I_error[u,v]
        
        delta_p =np.array(hessian_inverse @ H) 
        W = W @ np.linalg.pinv(np.array([[delta_p[0] + 1, delta_p[1], delta_p[2]], [delta_p[3], delta_p[4] + 1, delta_p[5]], [ 0, 0, 1]]))
   
    A_align = W
    err_list = np.asarray(err_list)

    return A_align, err_list


def track_multi_frames(template, img_list):
    A_list=[]
    template_list = []
    template_list.append(template)

    for num in range(len(img_list)):
        x1,x2 =  find_match(template, img_list[num])

        A = align_image_using_feature(x1, x2, 30, 1000)
        A_aligned, err_list = align_image(template, img_list[num], A)
        A_list.append(A_aligned)
        
        template =  np.array(warp_image(img_list[num], A_aligned, template.shape), dtype='uint8')
        template_list.append(template)
        
    return A_list


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 30
    ransac_iter = 1000

    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr, img_h = 500)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


