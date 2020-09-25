# Model
This is the repository holding the model backing the Wingman Jr. Addon.
There are different versions of the model for different releases; see each subfolder for different releases.

# Dataset
However, the underlying dataset format remained largely the same, but simply grew in size.
The dataset is >100K still images. Currently no moving images have been incorporated.
Unlike some datasets, many of the images have been collected by intercepting images from actual browsing sessions.
This helps keep the dataset grounded to its actual usage: browsing the internet.
The grading scheme is by nature subjective, but generally defines four classes.
* Safe (S)
* Questionable (Q)
* Racy (R)
* Explicit (X)

As the names would imply, the focus has been primarily on filtering of sexualized content.
However, some violent/horror/"scary" content has also been captured, and to a lesser degree, drug usage.

Safe - Images that are not sexualized or scary.
Questionable - Content that may be ambiguously or lightly sexualized, non-graphic spoofed sexualization, or slightly scary.
Racy - Soft pornography. Horror.
Explicit - Hard pornography. Extremely graphic violence.

Unfortunately due to the nature of the dataset it cannot be made public.