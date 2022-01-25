import matplotlib.pyplot as plt
import datasets
import torch.utils.data as data_utils
import torch


def dataset_preview(name, dataset_size, dataloader):

    if ds_name == "breast_cancer":
        nrows, ncols = 10, 6
        figsize = (14, 20)
    elif ds_name == "colon_cancer":
        nrows, ncols = 11, 9
        figsize = (18, 22)
    else:
        raise NotImplementedError

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, batch in zip(axs.flatten(), dataloader):
        batch = [x.squeeze() for x in batch]
        _, label, img = batch

        ax.imshow(img)
        if label:
            plt.setp(ax.spines.values(), color="red", linewidth=3)
        else:
            plt.setp(ax.spines.values(), color="green", linewidth=3)
        ax.tick_params(axis="both",
                       which="both",
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labeltop=False,
                       labelleft=False,
                       labelright=False)

    # turn off unused axes completely
    for ax in axs.flatten()[dataset_size:]:
        ax.set_axis_off()

    fig.tight_layout()
    plt.savefig(f"{name}_preview.png")
    print(f"saved {name}_preview.png")


def bag_preview(ds_name, bag):

    if ds_name == "breast_cancer":
        nrows, ncols = 24, 28
        figsize = (28, 24)
    elif ds_name == "colon_cancer":
        nrows, ncols = 8, 7
        figsize = (7, 8)
    else:
        raise NotImplementedError

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, (ax, patch) in enumerate(zip(axs.flatten(), bag)):

        black_patch = torch.zeros((32, 32, 3))
        if discard_patch(patch):
            ax.imshow(black_patch)
        else:
            ax.imshow(patch.permute(1, 2, 0))

        ax.tick_params(axis="both",
                       which="both",
                       bottom=False,
                       top=False,
                       left=False,
                       right=False,
                       labelbottom=False,
                       labeltop=False,
                       labelleft=False,
                       labelright=False)

    # turn off unused axes completely
    for ax in axs.flatten()[len(bag):]:
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    plt.savefig(f"{ds_name}_bag_preview.png")
    print(f"saved {ds_name}_bag_preview.png")


def discard_patch(patch, ratio=0.75, color_threshold=0.9):

    discard_threshold = torch.numel(patch) * ratio
    mask = patch.ge(color_threshold)

    if mask.sum() > discard_threshold:
        return True
    else:
        return False


if __name__ == "__main__":
    ds_names = ["breast_cancer"]
    for ds_name in ds_names:

        ds_p, ds_a = datasets.get_datasets(ds_name, shuffle_bag=False, keep_imgs=True)
        dl = data_utils.DataLoader(ds_p, shuffle=False)
        dataset_preview(ds_name, len(ds_p), dl)
        bag_preview(ds_name, ds_p[10][0])
