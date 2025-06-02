import sys
import gc
import os
import torch
import json
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionSafetyChecker,
)

sys.path.append("..")
sys.path.append("../third_party/TransformerLens")
from general_utils import set_deterministic

HF_TOKEN = ""


class SafteyChecker(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, clip_input, images):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts


def generate_with_seed(
    sd_pipeline,
    prompts,
    seed,
    output_path="./",
    image_path=None,
    image_params="",
    save_image=True,
):
    set_deterministic(seed)
    outputs = []
    for prompt in prompts:
        print(prompt)
        image = sd_pipeline(prompt)["images"][0]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if image_params != "":
            image_params = "_" + image_params

        image_name = image_path or f"{output_path}/{prompt}.jpeg"
        if save_image:
            print(image_name)
            image.save(image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def generate_images(sad_prompts, happy_prompts, neutral_prompts, output_path, seed):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    images = []
    for ind, text in enumerate(sad_prompts):
        image_path = f"{output_path}/sad_{ind}.jpeg"
        if not os.path.exists(image_path):
            prompt = f"{text}. Sad, devastating, crying, terrible. High resolution photography."
            generate_with_seed(
                pipe, [prompt], seed, image_path=image_path, save_image=True
            )
        images.append([text, "sad", f"sad_{ind}.jpeg"])

    for ind, text in enumerate(happy_prompts):
        image_path = f"{output_path}/happy_{ind}.jpeg"
        if not os.path.exists(image_path):
            prompt = f"{text}. Happy and uplifting. High resolution photography."
            generate_with_seed(
                pipe, [prompt], seed, image_path=image_path, save_image=True
            )
        images.append([text, "happy", f"happy_{ind}.jpeg"])

    for ind, text in enumerate(neutral_prompts):
        image_path = f"{output_path}/neutral_{ind}.jpeg"
        if not os.path.exists(image_path):
            prompt = f"{text}. High resolution photography."
            generate_with_seed(
                pipe, [prompt], seed, image_path=image_path, save_image=True
            )
        images.append([text, "neutral", f"neutral_{ind}.jpeg"])

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return images


def save_verified_images(images, output_path):
    verified_data = []
    for index, (text, sentiment, image_path) in enumerate(images):
        print("*" * 100)

        if sentiment == "happy":
            answerKey = 0
        elif sentiment == "sad":
            answerKey = 1
        elif sentiment == "neutral":
            answerKey = 2
        else:
            raise ValueError(f"Sentiment {sentiment} not recognized")

        if os.path.exists(f"./{output_path}/{image_path}"):
            sample = dict(
                image_path=image_path,
                text_data=text,
                sentiment=sentiment,
                choices=dict(label=["A", "B", "C"], text=["Happy", "Sad", "Neutral"]),
                answerKey=answerKey,
            )
            verified_data.append(sample)
            with open(f"./sentiment_vl.json", "w") as f:
                json.dump(verified_data, f)
            print(f"saved {len(verified_data)} verified data")
    return verified_data


def evaluate_dataset(data=None):
    if not data:
        with open("./sentiment_vl.json", "r") as f:
            data = json.load(f)

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    vl_verification_prompt_template = (
        "<|image|>. Is this scene happy, sad, or neutral? Answer in a single word:"
    )
    l_verification_prompt_template = (
        "{scene}. Is this scene happy, sad, or neutral? Answer in a single word:"
    )

    l_acc = 0
    vl_acc = 0

    answer_tokens = processor(
        text=["happy", "sad", "neutral"], return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    print(answer_tokens)
    for sample in data:
        sentiment = sample["sentiment"]
        text = sample["text_data"]
        print(sentiment, text, flush=True)

        l_verification_prompt = l_verification_prompt_template.format(scene=text)
        inputs = processor(
            text=l_verification_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(
            **inputs,
        )
        logits = outputs.logits[:, -1]
        scores = torch.softmax(logits, dim=-1).detach().cpu()
        scores = scores.squeeze()
        top_token = torch.argmax(scores).item()
        correct = top_token in answer_tokens[sentiment]
        print(
            "l: ", sentiment, processor.tokenizer.decode(top_token), correct, flush=True
        )
        if correct:
            l_acc += 1

        image = Image.open(
            f"./sentiment_analysis/verified_images/{sample['image_path']}"
        )
        print(image, flush=True)
        inputs = processor(
            image,
            vl_verification_prompt_template,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(
            **inputs,
        )
        logits = outputs.logits[:, -1]
        scores = torch.softmax(logits, dim=-1).detach().cpu()
        scores = scores.squeeze()
        top_token = torch.argmax(scores).item()
        correct = top_token in answer_tokens[sentiment]
        print(
            "vl: ",
            sentiment,
            processor.tokenizer.decode(top_token),
            correct,
            flush=True,
        )
        if correct:
            vl_acc += 1

    l_accuracy = l_acc / len(data)
    vl_accuracy = vl_acc / len(data)
    print(l_accuracy, vl_accuracy, flush=True)
    return l_accuracy, vl_accuracy


def main():
    seed = 42
    set_deterministic(seed)
    output_path = "./sentiment_analysis/images/"
    verified_output_path = "./sentiment_analysis/verified_images/"

    sad = [
        "A child sobs alone in the rain, clutching a torn letter, eyes searching the stormy sky for comfort.",
        "A grieving mother kneels by a fresh grave, rain pouring down as she whispers forgotten lullabies to the soil.",
        "A soldier’s farewell letter lies beside a folded flag, unread, as wind rattles the windows of an empty home.",
        "A starving stray dog shivers in the cold, eyes pleading for help no one is willing to give tonight.",
        "A lonely old man wipes away tears, staring at his late wife's picture, remembering their final words and promises.",
        "A broken swing sways in an empty playground, echoes of laughter long gone, now replaced by haunting silence.",
        "A girl sits by a hospital bed, holding her father’s lifeless hand, the machines already silent and still.",
        "A forgotten birthday cake melts away, candles unlit, the room eerily silent, decorations sagging with time and sorrow.",
        "A couple stands in the storm, one turning away forever, their hands parting like pages in a tragedy.",
        "A shivering homeless man curls up in a dark alley, unseen and forgotten, surrounded by the stench of despair.",
        "A blood-stained teddy bear lies in the mud, rain falling steadily, a child’s scream still echoing in memory.",
        "A coffin lowers into the ground as mourners stand silently in black, their tears mixing with the falling rain.",
        "A woman screams alone in a dark hallway, clutching a death certificate, the silence pressing in like a scream.",
        "A suicide note sits beside a bottle of pills on a nightstand, the clock ticking louder with each breath.",
        "A house burns in the distance, a child watching from the roadside, crying, clutching only a stuffed rabbit.",
        "A morgue drawer is pulled open, the toe tag swinging slightly in the cold air of the silent chamber.",
        "A man stares at a noose hanging in a dimly lit basement, his shadow trembling on the concrete wall.",
        "A crow picks at a wilted bouquet left on an overgrown grave, its caws lost in the wind’s cry.",
        "A wedding dress floats in a murky river, torn and stained, once white, now forgotten beneath the drifting fog.",
        "A crash site smolders at dawn, belongings scattered, a single shoe in the dirt, untouched by rescuers' hesitant steps.",
        "A mother clutches her stillborn baby, tears mixing with sweat and silence, whispering a name never spoken aloud.",
        "An empty crib remains untouched, dust collecting on the mobile above, silence louder than lullabies once imagined.",
        "A teenager's final text glows on a cracked phone screen in the rain, unread, trembling in the stormy light.",
        "A hearse drives alone down a foggy road, headlights barely piercing the gloom, a final journey without farewell.",
        "A man screams into a pillow, surrounded by empty liquor bottles and darkness, lost in a night without mercy.",
        "A wedding photo burns slowly in a fireplace, edges curling into ash, love erased by flame and quiet grief.",
        "A body lies covered in a white sheet as police tape flutters nearby, sirens fading into the distant silence.",
        "An abandoned asylum hallway echoes with faint footsteps and distant sobbing, the air thick with forgotten screams and pain.",
        "A child’s shoe sits at the edge of a frozen lake, untouched, a question hanging heavy in the cold.",
        "A girl lights candles around a gravestone on what would’ve been prom night, her dress soaked with quiet sorrow.",
        "A young boy hugs an urn, too small to understand the loss, whispering secrets to the ashes inside.",
        "A funeral procession moves through the rain, black umbrellas like wilted flowers, mourners silent beneath the weight of goodbye.",
        "A heart monitor flatlines, its final beep echoing through a sterile hospital room, death declared without a single word.",
        "A wheelchair sits empty at the edge of a cliff, facing the sea, wheels still, silence howling in the wind.",
        "A burned-out car smolders in the distance, family photos scattered in the grass, a story erased by smoke and ash.",
        "A man cradles his lifeless dog beneath a streetlamp on a rainy night, whispering thank you through trembling lips.",
        "A baby blanket lies abandoned in the rubble of a collapsed building, soot-stained memories tangled in cotton and grief.",
        "A girl stares at a missing poster with her sister’s face on it, heart pounding with fear and fading hope.",
        "A blood-soaked wedding dress is found in a forest clearing, surrounded by silence and the scent of decay.",
        "A soldier’s helmet lies forgotten in a trench, half-buried in mud and time, untouched since the final battle.",
        "A child's drawing is taped to a morgue door, marked with a name, a date, and broken-hearted scribbles.",
        "A mother leaves flowers on a roadside memorial, the wind blowing her hair across tear-streaked cheeks and trembling lips.",
        "A voicemail plays on repeat in a dark room, a final goodbye no one answered or ever responded to.",
        "A teen’s journal lies open to a page that ends mid-sentence, the ink smudged by unseen tears and silence.",
        "A ghost bike chained to a pole is covered in rain and wilted flowers, untouched by time or passerby glances.",
        "A man reads a letter that begins with 'I'm sorry' and ends in silence, tears soaking the fragile paper.",
        "A pair of shoes sits neatly by a bridge railing, untouched in the morning mist and ghostly pre-dawn light.",
        "A child draws stick figures of their family—one figure is crossed out in red crayon, forgotten but not forgiven.",
        "A birthday balloon floats into the sky, released by hands that couldn’t celebrate through the weight of remembered grief.",
        "A dying tree stands alone in a graveyard, branches reaching like bony fingers toward the sky, whispering names of the dead.",
        "A child stares through a rain-streaked window, clutching a stuffed animal tightly, waiting for someone who will never return home.",
        "An elderly man sits alone at a bus stop, no one comes to meet him, just the empty silence surrounding everything.",
        "A grave covered in fresh flowers under gray skies and silent mourners, each petal a goodbye unspoken and forever aching.",
        "A single shoe lies forgotten in the middle of a rainy intersection, untouched by any passing cars.",
        "A hospital hallway stretches endlessly, lit by flickering fluorescent lights, echoing the emptiness of the sterile walls.",
        "A girl weeps beside a broken photo frame, shards scattered on the floor, as memories flood her heart.",
        "A suicide note lies on a desk next to an untouched coffee cup, the room heavy with silence.",
        "A child lights a candle at a roadside memorial, face full of quiet grief, as the world moves on.",
        "A woman kneels beside a grave marked with her child’s name and a tiny teddy bear, eyes full of sorrow.",
        "An old wedding dress hangs in an abandoned house, gathering dust and memories, forgotten by time’s passing hand.",
        "A soldier’s uniform lies folded beside a flag and a framed photo, untouched by the hands of a loved one.",
        "A single wilted rose rests on a casket as rain begins to fall, soaking the earth with sorrow.",
        "A man screams into the night, alone on a deserted bridge, the sound swallowed by the dark silence.",
        "An empty crib in a nursery, untouched toys lining the shelves, a reminder of a life that was never lived.",
        "A bloodstained journal lies open on a kitchen floor in silence, its pages filled with lost hopes and pain.",
        "A lone swing creaks in the wind on a forgotten playground, its chains swaying like the hands of time.",
        "A figure sits at the edge of a pier, shoes off, staring at the black water, lost in thought.",
        "A dog waits outside a burned home, tail still, eyes watching, longing for the return of its lost family.",
        "A woman collapses to her knees in front of a closed hospital door, her sobs echoing in the empty hallway.",
        "A wedding photo lies torn in half on a dark bedroom floor, memories scattered like broken pieces of her heart.",
        "A man walks into the sea at dusk, waves swallowing his footprints, the horizon blurred by his silent tears.",
        "A child holds a candle vigil for a classmate who never came back, the flame flickering in the cold wind.",
        "An urn sits on a windowsill, untouched by the dust of months, a silent testament to a lost soul.",
        "A shattered mirror reflects a tear-streaked face and broken expression, pieces of glass scattered like her shattered dreams.",
        "A teenager’s backpack rests against a wall covered in missing person posters, her absence weighing heavy on the air.",
        "A girl clutches a voicemail recording on repeat, sobbing in darkness, as the silence between her tears grows louder.",
        "A bouquet lies at the feet of a statue, card reading 'Gone too soon,' surrounded by fallen petals and memories.",
        "A woman holds an ultrasound image, crying in an empty waiting room, the future she dreamed of slipping away.",
        "A father weeps silently at the site of a car crash memorial, his heart aching with the loss of his child.",
        "A family photo floats in floodwater inside a crumbling, abandoned home, the memories now submerged in sorrow and decay.",
        "A tree is carved with initials, now cracked and forgotten in an overgrown field, a symbol of lost love and time.",
        "A streetlight flickers over a figure curled on a park bench in the rain, its dim light failing to offer comfort.",
        "A man opens a letter with shaking hands and drops to the ground, the words inside shattering his world forever.",
        "A girl releases a balloon into the sky with her eyes closed, lips trembling, as a final goodbye slips away.",
        "A suicide hotline flyer blows across a train station platform in the wind, unnoticed by the hurried crowd around it.",
        "An empty wheelchair faces a sunset from a quiet cliffside overlook, the wind whispering through the trees.",
        "A boy stares at his reflection in a cracked window, eyes hollow, lost in the pain of his thoughts.",
        "A once-cheerful bedroom sits untouched since the accident, toys covered in dust, the air thick with silence.",
        "A wedding ring lies abandoned on a headstone, glinting in the rain, a symbol of lost love and grief.",
        "A man plays a sorrowful tune on a violin in an empty subway tunnel, his music echoing in the dark.",
        "A woman lies motionless in a bathtub, water still running over the edge, her mind overwhelmed with sorrow.",
        "A soldier’s boots sit beside a helmet and dog tags in a field of crosses, the wind softly blowing.",
        "A bloodied pillow lies on the floor of a silent bedroom, stained with the pain of a lost battle.",
        "A girl covers her ears as her parents scream in the next room, her heart pounding in fear.",
        "A dog lies beside its owner’s grave, refusing to move, loyal even in death, in complete stillness.",
        "A child hides beneath a table, eyes wide, clutching a broken toy, surrounded by an eerie quiet.",
        "A man watches old home videos alone, tears streaming down his cheeks, memories of better days haunting him.",
        "A burned photo album is all that remains after the fire, charred edges whispering of past memories lost.",
        "A graveyard shrouded in fog, one lone figure standing in the mist, lost in mourning and solitude.",
        "A mother folds a tiny outfit, tears falling onto the fabric, remembering a child she’ll never hold again.",
        "An engagement ring lies at the bottom of a wine glass, untouched, a symbol of a love that faded.",
        "A figure disappears into a storm, the wind howling like screams, swallowed by the darkness of the night.",
        "A child places a drawing on a gravestone and walks away in silence, the wind carrying away her hopes.",
        "A woman holds her head in her hands beside a freshly dug grave, her grief overwhelming and deep.",
        "A chair rocks slowly on a porch, no one left to sit in it, the evening breeze whispering.",
        "A voicemail plays again and again in a dark apartment, unanswered forever, echoing in the silence.",
        "A man writes a name in the condensation of a train window, eyes full of longing and regret.",
        "A field of white crosses stretches into the horizon, each marked with a name and a date, a solemn tribute.",
        "A girl lights birthday candles for a sibling who never made it to ten, her hands trembling slightly.",
        "A father hugs a pillow shaped like his daughter, lost too soon, tears soaking into the fabric.",
        "A photo of happier days sits beside a bottle of sleeping pills, untouched, the memories fading slowly.",
    ]

    happy = [
        "A child laughs joyfully, running through a field of sunflowers under a bright blue sky, feeling free.",
        "A couple shares a kiss under a fireworks-lit sky, celebrating New Year's Eve together with laughter and love.",
        "A golden retriever excitedly chases a ball across a sunny park, tail wagging, bringing joy to all nearby.",
        "Friends cheer, clinking glasses at a rooftop party, city lights sparkling behind them, creating lasting memories together.",
        "A grandmother hugs her grandchild tightly, both smiling with pure happiness, enjoying a peaceful moment of love.",
        "A young couple dances barefoot on the beach, waves crashing around them, their laughter filling the evening air.",
        "A child excitedly blows out birthday candles, surrounded by family and balloons, making a wish with all their heart.",
        "A musician plays guitar in the street, strangers clapping and dancing to the rhythm, creating an atmosphere of joy.",
        "A newlywed couple walks hand in hand through a garden filled with blooming roses, their hearts full of love.",
        "A father lifts his giggling daughter onto his shoulders at a summer carnival, both enjoying the joyful moments.",
        "A child runs through a field of daisies, laughter echoing in the sunshine, feeling the warmth of the day.",
        "A couple shares their first kiss as newlyweds under a glowing sunset, their love marking the beginning of forever.",
        "Friends toss colorful powder into the air at a vibrant Holi celebration, smiles and laughter spreading everywhere.",
        "A golden retriever leaps into a lake, kids cheering on the shore, the moment filled with pure excitement.",
        "A grandmother blows out 90 candles as her family sings joyfully around her, a lifetime of memories shared.",
        "A surprise party erupts in cheers as the birthday girl walks in, her heart filled with happiness and love.",
        "A father swings his son in the air on a beach at sunset, both enjoying the golden glow of evening.",
        "A baby giggles wildly during their first time on a swing, a joyous moment that melts everyone’s heart.",
        "A rainbow arcs over a quiet countryside after a warm summer rain, casting a colorful glow over the landscape.",
        "Children chase butterflies in a garden full of blooming tulips and sunflowers, laughing and running in pure joy.",
        "A young couple dances barefoot in their living room, music playing softly, their hearts filled with happiness and love.",
        "Friends laugh around a bonfire, roasting marshmallows under the starry night sky, sharing stories and creating memories together.",
        "A classroom of kids cheers as balloons fall for the last day of school, their excitement palpable as summer begins.",
        "Two best friends take silly selfies surrounded by confetti and balloons, laughing together as they capture the moment forever.",
        "A woman receives a marriage proposal beneath a canopy of fairy lights, her heart racing with joy and surprise.",
        "Siblings build the tallest sandcastle on a sunny, perfect beach day, their laughter mixing with the sound of crashing waves.",
        "A bride and groom run through a tunnel of sparklers after their ceremony, hand in hand, glowing with happiness.",
        "A farmer smiles, holding a basket overflowing with freshly picked vegetables, feeling proud of the day’s hard work.",
        "A child hugs their puppy tightly, both grinning with joy, their hearts full of unconditional love and friendship.",
        "A couple watches the sunrise wrapped in a blanket on a mountaintop, feeling the peace of the early morning light.",
        "A girl spins in a new dress as cherry blossoms fall around her, her face glowing with pure delight and excitement.",
        "A teenager jumps with joy after receiving their college acceptance letter, overwhelmed with happiness and the promise of a new chapter.",
        "A marching band parades down a street, music echoing and people clapping, celebrating a day of tradition and community spirit.",
        "A mom and daughter bake cookies together, laughing as flour flies everywhere, enjoying a special moment in the kitchen.",
        "Friends jump into a lake together, water splashing under the summer sun, their laughter ringing out across the water.",
        "A couple paints their new home, smiling with splatters of color on their clothes, creating their dream space together.",
        "Kids play tag through a sprinkler on a hot summer day, their laughter filling the air as water sprays everywhere.",
        "A crowd claps as a street performer nails a backflip, impressed by the daring trick and cheering enthusiastically.",
        "A birthday cake covered in candles glows as the room sings in unison, celebrating another year of life and love.",
        "A toddler hugs a balloon animal, face lit up with pure joy, their eyes sparkling with excitement and wonder.",
        "A boy wins a prize at the fair, holding up a giant teddy bear, his face beaming with pride and excitement.",
        "Grandparents dance together at their 50th anniversary, surrounded by generations of family, their love still as strong as ever.",
        "A girl tosses her graduation cap high, beaming with pride, surrounded by friends and family celebrating her accomplishment.",
        "Children sit in a circle, sharing stories and snacks on a sunny field trip, enjoying each other's company.",
        "A couple shares ice cream on a warm summer night, laughing together under the starry sky, savoring the moment.",
        "A small child opens a gift, eyes wide with wonder and excitement, unable to contain their joy and curiosity.",
        "Two friends high-five at the finish line of their first marathon, exhausted but elated by their achievement.",
        "Parents cheer as their baby takes their first steps in the living room, capturing the precious milestone on camera.",
        "A kite soars high in a bright blue sky, pulled by a running child whose laughter echoes in the air.",
        "A dog runs through a meadow, tongue out, tail wagging furiously, thrilled by the freedom and open space.",
        "A young boy gives his mom a handmade card for Mother’s Day, his face beaming with pride and love.",
        "Sunlight pours through the trees as a family hikes together in the forest, enjoying nature and each other's company.",
        "Laughter fills the air as friends pile into a photo booth at a wedding, creating memories to last a lifetime.",
        "A classroom erupts in cheers as the teacher announces a surprise pizza party, delighting the students with joy.",
        "A couple clinks glasses during a candlelit dinner on their anniversary, savoring the love and connection between them.",
        "A child proudly shows off a drawing taped to the fridge, eager to share their creativity with everyone in the house.",
        "Friends sing along to music on a road trip with the windows down, enjoying the carefree spirit of the journey.",
        "A garden bursts with color as butterflies flutter between the flowers, the air filled with the scent of blossoms.",
        "New parents cradle their newborn, eyes full of awe and love, marveling at the tiny miracle they created together.",
        "Kids laugh while carving pumpkins, their faces glowing in the autumn sun, enjoying the season's festive spirit.",
        "Snowflakes fall gently as children build a snowman, scarves and smiles in place, surrounded by a winter wonderland.",
        "A family cuddles together on the couch, watching their favorite movie, wrapped in warmth and shared joy.",
        "A teacher smiles as a student hands them a flower and thank you note, grateful for the kindness shown.",
        "A toddler dances with delight at a backyard bubble party, joyfully chasing after the floating bubbles in the air.",
        "Fireworks light up the sky above a cheering crowd on Independence Day, marking the celebration with explosive beauty.",
        "A birthday piñata explodes, candy raining down on screaming, happy children, who scramble excitedly to collect the treats.",
        "Siblings exchange gifts and hugs during a cozy winter holiday morning, sharing the warmth of family love and happiness.",
        "A young girl proudly rides her bike without training wheels for the first time, beaming with accomplishment and joy.",
        "Friends picnic by a lake, passing food and sharing stories under the sun, enjoying the peacefulness of the outdoors.",
        "A man surprises his partner with breakfast in bed and fresh flowers, starting their day with love and care."
        "Kids giggle while playing hide and seek in a sunflower field, their laughter echoing through the warm breeze.",
        "A couple watches dolphins leap from the ocean on a tropical vacation, the water sparkling under the sun.",
        "A young artist beams beside their painting at a local gallery show, proud of their creative achievement.",
        "A woman jumps into her best friend's arms at the airport reunion, both overcome with joy and excitement.",
        "A family gathers around a piano, singing holiday songs together, sharing the warmth of their love and tradition.",
        "A choir performs joyfully in the town square during a festival, their voices harmonizing with the festive atmosphere.",
        "A young boy plants a tree, proud smile on his dirt-covered face, knowing he is helping the planet.",
        "A street is filled with colorful floats and dancers in a local parade, the air buzzing with excitement.",
        "A toddler claps to music at their first birthday party, completely entranced by the fun and energy around them.",
        "A newly adopted dog licks their owner's face, tail wagging nonstop, excited to begin their new life together.",
        "A couple holds hands on a park bench, laughing at old memories, savoring the quiet moments together.",
        "Best friends exchange handmade friendship bracelets under a blooming cherry tree, giggling at the beauty of the day.",
        "A team celebrates a last-minute goal, hugging on the soccer field, overwhelmed by the excitement and victory.",
        "A teen proudly shows their freshly baked cake to their grandparents, beaming with pride over their culinary success.",
        "A man twirls his partner in the middle of a crowded festival, their laughter blending with the lively music.",
        "A girl hugs a giant stuffed animal she won at the carnival, smiling with pure joy and accomplishment.",
        "A group of campers roast marshmallows and sing songs around the fire, sharing stories under the starlit sky.",
        "A boy flies his paper airplane from a hilltop, smiling wide as it glides gracefully through the air.",
        "A grandmother teaches her grandchild how to knit beside a sunny window, both enjoying the quiet afternoon together.",
        "A couple laughs uncontrollably during their wedding vows, overwhelmed by the love and joy of the moment.",
        "A dad builds a treehouse while his kids hand him nails, their faces lighting up with excitement and joy.",
        "Two friends leap off a dock into the lake, midair smiles frozen in time, enjoying the thrill of summer.",
        "A teenager hugs their sibling after opening a long-wished-for holiday gift, both thrilled by the thoughtful surprise.",
        "A choir of children sings carols door to door, candles in hand, spreading holiday cheer and warmth to all.",
        "A woman celebrates her promotion with a champagne toast and happy tears, surrounded by friends and family.",
        "A father and daughter stargaze in the backyard with a telescope, marveling at the beauty of the night sky.",
        "A crowd erupts as a young violinist finishes their first solo recital, the audience applauding their remarkable performance.",
        "A baby claps as bubbles float all around them in the sunlight, their face lit up with pure delight.",
        "A group of coworkers celebrates a big win with confetti and cake, toasting to their success and teamwork.",
        "A couple laughs together while painting goofy designs on Easter eggs, enjoying the warmth of the holiday spirit.",
        "A family rides bicycles together through a blooming spring trail, the air fresh and filled with laughter.",
        "A child discovers the ocean for the first time, eyes wide with wonder as the waves crash at their feet.",
        "A boy hugs his favorite stuffed animal, surrounded by bedtime stories and stars, feeling safe and loved.",
        "Friends share milkshakes at a retro diner, laughing between sips, reminiscing about old memories from high school.",
        "A woman cheers as she crosses the finish line of her first 10K run, feeling exhausted yet proud.",
        "Parents hold hands, watching their kids perform on stage with beaming pride, their hearts full of joy.",
        "A beach picnic unfolds as waves crash and kids build sandcastles, the sun shining brightly over the scene.",
        "The sun rises behind a couple wrapped in blankets on a balcony, sharing a peaceful moment before the day begins.",
        "Friends dance barefoot on the grass as music plays into the evening, laughter filling the air under the stars.",
        "A child finds a four-leaf clover, grinning with delight, feeling like they’ve discovered a little piece of magic.",
        "A teacher hugs her class goodbye on the last day of school, knowing their students are ready for the summer break.",
        "A man proposes under twinkling lights, his partner covering their face in joy, overwhelmed with love and surprise.",
        "A toddler learns to clap, delighted by every sound they make, proudly showing off their new skill to everyone.",
        "A mom cheers as her daughter scores her first soccer goal, jumping up and down with pride in the stands.",
        "Two friends laugh uncontrollably, tears in their eyes, after an inside joke that no one else quite understands.",
        "A family sets off colorful lanterns into the night sky, making wishes together, their hearts full of hope.",
    ]

    neutral = [
        "A man waits at a bus stop, checking his watch occasionally, looking up every few moments as cars pass by.",
        "A bird perches on a windowsill, observing the quiet street below, its feathers ruffled by the gentle breeze.",
        "A woman organizes books on a shelf in a quiet library, carefully making sure they are in perfect order.",
        "A streetlamp stands tall in an empty parking lot at dusk, its light flickering softly as the sky darkens.",
        "Two people walk side by side without speaking, focused on the road ahead, each lost in their own thoughts.",
        "A leaf floats down a river, carried by a gentle current, drifting slowly toward the distant bend in the water.",
        "A car idles at a red light, rain tapping softly on the windshield, the world blurred by droplets.",
        "A worker sweeps the floor of a train station as people pass, the sound of his broom echoing in the quiet.",
        "A man types on a laptop in a dimly lit cafe, his fingers moving quickly over the keys as music plays softly.",
        "A pair of shoes sits neatly by the front door, their laces tied and waiting to be worn again.",
        "A dog lies on a porch, eyes half-closed, tail still, enjoying the warmth of the afternoon sun.",
        "A student stares at a whiteboard, pencil tapping rhythmically on their notebook, trying to focus on the lesson.",
        "A bicycle wheel spins slowly after the rider hops off, the movement gradually slowing as the bike comes to rest.",
        "A woman folds laundry methodically while a TV plays in the background, her movements steady and practiced.",
        "A train arrives on schedule, commuters stepping off one by one, each moving toward their destination with purpose.",
        "A pile of logs rests beside a shed in the backyard, stacked neatly in preparation for the upcoming winter.",
        "A man eats lunch alone in a cafeteria, scrolling through his phone, occasionally glancing up at his surroundings.",
        "A woman stands at a vending machine, deciding between snacks, her hand hovering over the options.",
        "A cat stretches on a windowsill as clouds drift past, lazily watching the changing sky with half-closed eyes.",
        "A stack of papers sits untouched on a desk in an office, waiting to be sorted through later in the day.",
        "A traffic cone marks a freshly painted line on the road, its bright color standing out against the wet pavement.",
        "A janitor mops a tiled floor under fluorescent lights, methodically working his way through the empty hallway.",
        "A woman waters plants on her balcony, looking down at the street, her thoughts drifting as she tends to her flowers.",
        "A delivery person knocks on a door and waits patiently, glancing around at the quiet neighborhood as they stand.",
        "A man adjusts his tie in front of a mirror before leaving, making sure everything is perfectly aligned.",
        "A chessboard sits mid-game in a park, no players nearby, the pieces waiting for someone to continue.",
        "A bus pulls away from a stop, leaving behind one waiting passenger, who watches it disappear into the distance.",
        "A parking garage slowly empties as evening sets in, the sound of cars driving off echoing in the quiet space.",
        "A dog barks in the distance as someone walks by with groceries, the sound lingering in the calm air.",
        "A construction crane looms over a half-finished building, its shadow casting across the surrounding construction site.",
        "A street cleaner passes by early in the morning, sweeping the gutters as the first rays of sunlight break through.",
        "A person flips through a magazine in a waiting room, glancing at the pages absentmindedly as time passes.",
        "A closed umbrella leans against a cafe table outside, abandoned for now as the rain has stopped.",
        "A man writes something on a whiteboard during a meeting, his hand moving quickly as others watch.",
        "A slow-moving river reflects the overcast sky and bare trees, the water rippling gently in the breeze.",
        "A boy kicks a soccer ball against a brick wall, watching it bounce back with a soft thud.",
        "A dog chases its tail briefly, then stops and lies down, resting on the cool grass in the shade.",
        "A woman stands at the photocopier, waiting as papers slide out, glancing at her watch occasionally.",
        "A man repairs a bicycle chain, tools scattered on the ground, carefully tightening the links one by one.",
        "A field of grass sways gently in the breeze, uninterrupted by anything except the sound of wind.",
        "A road sign points in two directions at a rural intersection, the landscape stretching out in each way.",
        "A delivery truck unloads boxes in front of a quiet store, the driver lifting each item with care.",
        "A pedestrian crosses the street with a grocery bag in each hand, walking briskly toward the sidewalk.",
        "A digital clock changes from 3:59 to 4:00 with a soft click, the room now bathed in dim light.",
        "A man eats noodles at a small corner restaurant, watching passersby through the foggy window.",
        "A street musician tunes his guitar, no audience yet, the sound of strings ringing through the air.",
        "A calendar page flutters slightly as the fan hums nearby, the date marking the start of a new week.",
        "A woman adjusts her scarf while walking past a construction site, eyeing the workers as they busy themselves.",
        "A man scribbles on a napkin in a quiet diner, his pen scratching against the paper as he thinks.",
        "A bird lands on a power line beside others already perched, its feathers ruffled by the wind.",
        "A train schedule scrolls across a screen at the station entrance, passengers glancing up at it occasionally.",
        "A woman sits on a bench reading a paperback novel, the pages turning slowly as she becomes engrossed.",
        "A squirrel pauses on a fence, tail twitching, then runs off, disappearing into the nearby trees.",
        "A barista pours coffee into a paper cup, steam rising steadily, filling the cafe with a rich aroma.",
        "A row of bicycles is parked in front of a school building, students milling about as the bell rings.",
        "A computer screen displays a loading icon in an empty office, the cursor spinning as the software loads.",
        "A person stands under a streetlight waiting for a rideshare, checking their phone for updates.",
        "A flag flutters outside a post office in a light breeze, its colors vivid against the clear sky.",
        "A man replaces a lightbulb in the ceiling of a stairwell, standing on a ladder to reach the socket.",
        "A barge moves slowly down the river, silent and steady, the water rippling gently as it passes.",
        "A hallway stretches in both directions, lit by evenly spaced lights, the floor polished and clean.",
        "A plane taxis along the runway, passengers still seated, waiting for the signal to take off.",
        "A woman ties her shoelaces before heading out for a walk, adjusting her jacket as she stands.",
        "A store clerk arranges cereal boxes neatly on a shelf, making sure they’re aligned with the others.",
        "A briefcase rests beside a bench in a quiet plaza, its owner standing nearby, talking on the phone.",
        "A cat walks across a keyboard and settles on a desk, curling up in a cozy spot by the monitor.",
        "A man sips tea while reading the newspaper at a kitchen table, the aroma filling the room.",
        "A car gets washed at an automatic station, soap covering the windows as water sprays all around.",
        "A woman scrolls through her phone in line at a pharmacy, occasionally glancing up as the line moves.",
        "A row of mailboxes stands at the edge of a gravel road, each one slightly askew from the wind.",
        "A clock ticks loudly in an otherwise silent room, marking the passage of time as nothing else happens.",
        "A man carries two bags of groceries up a stairwell, careful not to drop anything as he ascends.",
        "A cactus sits in the corner of a sunlit windowsill, its spines casting shadows across the desk.",
        "A painter tapes the edges of a wall before starting, preparing the space for a fresh coat of paint.",
        "A man locks his bicycle to a street pole before entering a store, making sure it’s secure.",
        "A vending machine glows in an otherwise dark hallway, its lights illuminating the snacks inside.",
        "A security guard patrols a shopping center, eyes scanning calmly, looking out for any signs of trouble.",
        "A jacket hangs from a hook in a sparsely decorated hallway, the fabric slightly wrinkled from use.",
        "A mechanic checks a car engine while a radio plays nearby, the soft hum of the music filling the air.",
        "A woman adjusts her glasses and returns to typing, her fingers moving swiftly over the keyboard.",
        "A child stares out the car window during a traffic jam, watching the passing cars with a bored expression.",
        "A group of seagulls circles above a quiet beach, their calls echoing as they glide through the air.",
        "A calendar hangs on the wall, flipped to the current month, the days marked with various events.",
        "A man closes the lid on his laptop and stands slowly, stretching as he prepares to leave.",
        "A broom leans against a wall outside a locked store, its bristles worn from years of use.",
        "A thermos steams beside a notebook on a wooden picnic table, the scent of hot coffee lingering in the air.",
        "A woman places items carefully on the checkout conveyor belt, making sure everything is in order.",
        "A train conductor checks tickets as the train rocks gently, moving toward its next stop in the city.",
        "A person flips pancakes in a pan on a stovetop, the sizzling sound filling the kitchen.",
        "A warehouse worker scans boxes with a barcode reader, each item being logged in for inventory.",
        "A parking meter blinks red as the timer runs out, signaling the end of the allotted parking time.",
        "A drone buzzes quietly above a grassy field, hovering just above the ground as it surveys the area.",
        "A stack of folded towels rests on a hotel room bed, neatly arranged for the next guest.",
        "A dog looks out a window as people walk by, its eyes following their movements with curiosity.",
        "A newspaper lies folded on a bench at a train platform, waiting to be picked up by someone passing by.",
        "A man walks through a revolving door into a tall office building, his footsteps echoing in the lobby.",
        "A plane leaves a contrail across a pale blue sky, leaving behind a thin white streak as it ascends.",
        "A coffee mug sits on a desk beside a laptop, steam rising gently from the hot drink.",
        "A bus pulls into a station, doors opening with a hiss, passengers stepping off and others boarding.",
        "A shoelace dangles untied as someone walks up a flight of stairs, the end swaying with each step.",
        "A pair of sunglasses rests on a dashboard, sun pouring through the window and casting shadows on the seat.",
        "A printer hums in the corner of a quiet classroom, its pages sliding out one by one.",
    ]

    print("Generating sentiment analysis images..")
    all_images = generate_images(sad, happy, neutral, output_path, seed)
    gc.collect()
    torch.cuda.empty_cache()
    print("** Manually verify images in <output_path> and delete bad generations **")
    saved_images = save_verified_images(all_images, verified_output_path)
    print("Evaluating..")
    l_accuracy, vl_accuracy = evaluate_dataset(saved_images)
    print("Accuracy on the langugage and vision datasets: ", l_accuracy, vl_accuracy)


if __name__ == "__main__":
    main()
