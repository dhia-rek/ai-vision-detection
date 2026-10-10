# Sample credits

## objects/

Wikimedia Commons, **CC0 1.0**, downscaled to 960 px wide. No identifiable people.

| File | Source |
| --- | --- |
| `objects/bicycles.jpg` | [Bicycles parked in Amsterdam](https://commons.wikimedia.org/wiki/File:Bicycles_parked_in_Amsterdam.jpg) |
| `objects/cats.jpg` | [Black and white cats lying down on steps](https://commons.wikimedia.org/wiki/File:Black_and_white_cats_lying_down_on_steps.jpg) |
| `objects/dog.jpg` | [Border collie lying on the grass](https://commons.wikimedia.org/wiki/File:Border_collie_lying_on_the_grass.jpg) |
| `objects/livingroom.jpg` | [Living room in apartment of Condomínio do Edifício Zaher, Le Blond, Rio de Janeiro, Brazil](https://commons.wikimedia.org/wiki/File:Living_room_in_apartment_of_Condom%C3%ADnio_do_Edif%C3%ADcio_Zaher,_Le_Blond,_Rio_de_Janeiro,_Brazil.jpg) |
| `objects/street.jpg` | [Jeep parked on bicycle lane in Ebisu](https://commons.wikimedia.org/wiki/File:Jeep_parked_on_bicycle_lane_in_Ebisu.jpg) (one distant, non-identifiable pedestrian) |

## faces/

Same files as `tests/fixtures/faces/` (adult models / public-domain government photo);
sources, authors and licences (CC0, public domain) are listed in
[tests/fixtures/faces/README.md](../../tests/fixtures/faces/README.md).

## sign/

Same files as in `tests/fixtures/sign/`: letter photos from the Hugging Face dataset
EitanG98/asl_letters (Unlicense) and gesture photos from Wikimedia Commons (CC0 / public
domain). Hands only. Sources per file: [tests/fixtures/sign/README.md](../../tests/fixtures/sign/README.md).

## video/

Slideshow clips generated from the still photos above (not real recordings), H.264:

| File | Built from |
| --- | --- |
| `fingerspelling_cab.mp4` | `letter_C`, `letter_A`, `letter_B` (1.5 s each) then 2 s blank |
| `faces.mp4` | `happy`, `angry`, `neutral` (2 s each, 640x480 letterboxed) |
