'''
Convert raw model output into usable objects.

Postprocessing converts:
tensor output into:
{
   bbox:
   confidence:
   class_name:
}

Responsibilities:
- Confidence filtering
- Bounding box extraction
'''

