from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from tqdm import tqdm

from mtgproxies.plotting import SplitPages

from PIL import Image, ImageDraw

image_size = np.array([745, 1040])


def _occupied_space(cardsize, pos, border_crop: int, closed: bool = False):
    return cardsize * (pos * image_size - np.clip(2 * pos - 1 - closed, 0, None) * border_crop) / image_size


def print_cards_matplotlib(
    images,
    filepath,
    papersize=np.array([8.27, 11.69]),
    cardsize=np.array([2.5, 3.5]),
    border_crop: int = 14,
    interpolation="lanczos",
    dpi: int = 600,
    background_color=None,
):
    """Print a list of cards to a pdf file.

    Args:
        images: List of image files
        filepath: Name of the pdf file
        papersize: Size of the paper in inches. Defaults to A4.
        cardsize: Size of a card in inches.
        border_crop: How many pixel to crop from the border of each card.
    """
    # Cards per figure
    N = np.floor(papersize / cardsize).astype(int)
    if N[0] == 0 or N[1] == 0:
        raise ValueError(f"Paper size too small: {papersize}")
    offset = (papersize - _occupied_space(cardsize, N, border_crop, closed=True)) / 2

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Choose pdf of image saver
    if filepath[-4:] == ".pdf":
        saver = PdfPages
    else:
        saver = SplitPages

    with saver(filepath) as saver, tqdm(total=len(images), desc="Plotting cards") as pbar:
        while len(images) > 0:
            fig = plt.figure(figsize=papersize)
            ax = fig.add_axes([0, 0, 1, 1])  # ax covers the whole figure
            #  Background
            if background_color is not None:
                plt.gca().add_patch(Rectangle((0, 0), 1, 1, color=background_color, zorder=-1000))

            for y in range(N[1]):
                for x in range(N[0]):
                    if len(images) > 0:
                        img = plt.imread(images.pop(0))

                        # Crop left and top if not on border of sheet
                        left = border_crop if x > 0 else 0
                        top = border_crop if y > 0 else 0
                        img = img[top:, left:]

                        # Compute extent
                        lower = (offset + _occupied_space(cardsize, np.array([x, y]), border_crop)) / papersize
                        upper = (
                            offset
                            + _occupied_space(cardsize, np.array([x, y]), border_crop)
                            + cardsize * (image_size - [left, top]) / image_size
                        ) / papersize
                        extent = [lower[0], upper[0], 1 - upper[1], 1 - lower[1]]  # flip y-axis

                        plt.imshow(
                            img,
                            extent=extent,
                            aspect=papersize[1] / papersize[0],
                            interpolation=interpolation,
                        )
                        pbar.update(1)

            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # Hide all axis ticks and labels
            ax.axis("off")

            saver.savefig(dpi=dpi)
            plt.close()


def print_cards_fpdf(
    images,
    filepath,
    papersize=np.array([210, 297]),
    cardsize=np.array([2.5 * 25.4, 3.5 * 25.4]),
    border_crop: int = 14,
    hSpace=0,
    vSpace=0,
    background_color: tuple[int, int, int] = None,
    intelligent_background: bool = False,
    cropmarks: bool = True,
    return_pdf: bool = False,
    queue = None,
):
    """Print a list of cards to a pdf file.

    Args:
        images: List of image files
        filepath: Name of the pdf file
        papersize: Size of the paper in inches. Defaults to A4.
        cardsize: Size of a card in inches.
        border_crop: How many pixel to crop from the border of each card.
    """
    from fpdf import FPDF

    # Cards per sheet
    N = np.floor(papersize / cardsize).astype(int)
    if N[0] == 0 or N[1] == 0:
        raise ValueError(f"Paper size too small: {papersize}")
    cards_per_sheet = np.prod(N)
    offset = (papersize - _occupied_space(cardsize, N, border_crop, closed=True)) / 2

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Initialize PDF
    pdf = FPDF(orientation="P", unit="mm", format="A4")

    queue.put(["message", "Plotting cards..."], timeout=5)
    for i, image in enumerate(tqdm(images, desc="Plotting cards")):
        queue.put(["progress", str(int((i+1)*50/len(images))+50)], timeout=5)
        if i % cards_per_sheet == 0:  # Startign a new sheet
            pdf.add_page()
            if background_color is not None:
                pdf.set_fill_color(*background_color)
                pdf.rect(0, 0, papersize[0], papersize[1], "F")

        x = (i % cards_per_sheet) % N[0]
        y = (i % cards_per_sheet) // N[0]

        # Crop left and top if not on border of sheet
        left = border_crop if x > 0 else 0
        top = border_crop if y > 0 else 0

        if left == 0 and top == 0:
            cropped_image = image
        else:
            path = Path(image)
            cropped_image = str(path.parent / (path.stem + f"_{left}_{top}" + path.suffix))
            if not Path(cropped_image).is_file():
                # Crop image
                plt.imsave(cropped_image, plt.imread(image)[top:, left:])

        # Compute spaces between images
        hBorder = np.full(N[0], 0)
        vBorder = np.full(N[1], 0)

        if hSpace is not 0:
            hBorder = np.arange(0, N[0]*hSpace, hSpace)
            hBorder = hBorder-hSpace*(N[0]-1)/2
        if vSpace is not 0:
            vBorder = np.arange(0, N[1]*vSpace, vSpace)
            vBorder = vBorder-vSpace*(N[1]-1)/2

        # Compute extent
        lower = offset + _occupied_space(cardsize, np.array([x, y]), border_crop) + np.array([hBorder[x], vBorder[y]])
        size = cardsize * (image_size - [left, top]) / image_size


        #Custom background
        if intelligent_background is True:
            boxsize = 40
            radius = 42
            path = Path(image)
            customBG_image = str(path.parent / ("customBG_" + path.stem + path.suffix))
            customBG_image_path = Path(customBG_image)

            im = Image.open(path).convert('RGBA')
            overlay = Image.new('RGBA', im.size)
            draw = ImageDraw.Draw(overlay)

            color_topleft = im.getpixel((20, 20))
            color_topright = im.getpixel((im.size[0]-20, 20))
            color_bottomleft = im.getpixel((20, im.size[1]-20))
            color_bottomright = im.getpixel((im.size[0]-20, im.size[1]-20))

            draw.rectangle([(0, 0), (boxsize,boxsize)], fill=color_topleft)
            draw.rectangle([(0, im.size[1]-boxsize), (boxsize, im.size[1])], fill=color_bottomleft)
            draw.rectangle([(im.size[0]-boxsize, 0), (im.size[0], boxsize)], fill=color_topright)
            draw.rectangle([(im.size[0]-boxsize, im.size[1]-boxsize), (im.size[0], im.size[1])], fill=color_bottomright)
            draw.rounded_rectangle([(0, 0), im.size], radius=radius, fill=(0, 0, 0, 0))

            im = Image.alpha_composite(im, overlay)
            im = im.convert("RGB") # Remove alpha for saving in jpg format.
            im.save(customBG_image_path)

            current_image = plt.imread(customBG_image_path)

            cropped_image = customBG_image

        # Plot image
        pdf.image(cropped_image, x=lower[0], y=lower[1], w=size[0], h=size[1])

        if cropmarks and ((i + 1) % cards_per_sheet == 0 or i + 1 == len(images)):
            # If this was the last card on a page, add crop marks
            pdf.set_line_width(0.05)
            pdf.set_draw_color(255, 255, 255)
            a = cardsize * (image_size - 2 * border_crop) / image_size
            b = papersize - N * a
            for x in range(N[0] + 1):
                for y in range(N[1] + 1):
                    mark = b / 2 + a * [x, y]
                    pdf.line(mark[0] - 0.5, mark[1], mark[0] + 0.5, mark[1])
                    pdf.line(mark[0], mark[1] - 0.5, mark[0], mark[1] + 0.5)

    tqdm.write(f"Writing to {filepath}")
    if(return_pdf == True):
        queue.put(["complete", "true"], timeout=5)
        return pdf

    pdf.output(filepath)
