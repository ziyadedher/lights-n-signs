from boto.mturk.connection import MTurkConnection
from boto.mturk.layoutparam import LayoutParameter
from boto.mturk.layoutparam import LayoutParameters
from drive import get_image_urls
import json

# 10mph
HIT_LAYOUT = "3BX02TL61NQB974KGIKHLR7UW1UBIE"
HIT_TYPE = "3CM00QMEULMPDN38Y6S0DUP25AWW6P"

def create_tasks(n):
    mtc = MTurkConnection(aws_access_key_id='AKIAJUXDOI7EZEMQMDEQ',
    aws_secret_access_key='hw8BWplLlTdhyFsbCuU6HYy1zAAfgznCP9Xuz8x0',
    host='mechanicalturk.amazonaws.com')
    tasks = []

    for (image, image_id) in get_image_urls(n):
        # Create your connection to MTurk
        image_url = LayoutParameter('image_url', image)
        obj_to_find = LayoutParameter('objects_to_find','Speed Limit 10 sign (do NOT include the sign pole, only the sign face)')
        params   = LayoutParameters([ image_url, obj_to_find ])
        response = mtc.create_hit(
          hit_layout    = HIT_LAYOUT,
          layout_params = params,
          hit_type      = HIT_TYPE,
        )

        # The response included several fields that will be helpful later
        hit_type_id = response[0].HITTypeId
        hit_id = response[0].HITId
        print("Your HIT has been created. You can see it at this link:")
        print("https://www.mturk.com/mturk/preview?groupId={}".format(hit_type_id))
        print("Your HIT ID is: {}".format(hit_id))

        tasks.append({"image_url": image, "image_id": image_id, "hit_id": hit_id})

    with open("annotations/mturk_10mph.json", "w+") as f:
        json.dump(tasks, f)

if __name__ == '__main__':
    create_tasks(2000)
