---
layout: post
title: "Anatomy Of A Domain Library"
date: 2021-06-22
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Anatomy Of A Domain Library

Hello, everyone, and welcome to the Pritchard Step Podcast.
Today, I wanna talk about the anatomy of the domain libraries.
that we also work on here at Pytorch, namely torch vision, torch audio, and torch text.
I'm not going to talk about the libraries, in particular, any one library in particular, but torture vision is definitely the most well developed and most featureful to main libraries.
So many of the things that I'm gonna say are based off of things that I know about torch vision.
Alright.
So here's a question.
Why do we have domain libraries? Like, why isn't PyTorch just one giant repository that contains, you know, how to do distributed computing and how to do operators and how to do profiling and tons and tons of stuff.
And then, you know, what's, you know, throwing in a little bit of, you know, image processing operators or, you know, text processing models? Like, you know, multi head attention is in Pyturg Core.
Why can't everything else be? And so there's a few reasons why the domain libraries exist as separate libraries from Pyturgical Core.
So one is that in particular domains like image and text and audio, they're often very domain specific gadgets.
These gadgets don't really make sense in any other context.
Like, for example, In vision, you need to have a JPEG decoder because you are commonly working with images that are in JPEG and you need them to be pixels so that you can start doing, you know, deep learning on them.
And it's would be pretty strange for PyTorch to come with a JPEG decoder and you know, wave decoder and, you know, every, you know, file format known under the sun.
So the domain libraries exist because there's a lot of extra stuff that you need to actually do work in one of these domains, but we just don't wanna keep shoveling everything into the main Pytors library because that makes your wheels bigger.
And it just it can get pretty out of control, especially because there are lots and lots of things you might wanna do.
So domain libraries give us an easy escape valve where we can say, oh yeah, you know, this stuff is great.
We wanna support it, but it just doesn't go in the main fighter's library It's gonna go in one of these extra libraries.
And, yes, sometimes there are operators that, like, you know, might be useful in multiple domains.
but usually it's pretty obvious where they should go.
The point about JPEG decoding is also another good point because another thing that you often need when you're working in a domain is there a bunch of other libraries that you might actually need, like f f m peg or live AV or pillow.
It's etcetera, etcetera.
And once again, it would be pretty suboptimal if, you know, when you installed Pytorch, it also got all these dependencies along.
So another good metric for, you know, should I make a domain library or, you know, should I not, is, you know, are there any dependencies you need? If there are no dependencies, while Pytorch might be a good place to put it because, you know, Pytorch tries to keep a very slim dependency set, only the bare minimum that you actually we actually even got rid of our numpai dependency in one dot nine.
This was accidental.
We didn't actually mean to do this.
But when people realize this is what what happened, okay.
Sorry.
We broke a bunch of people's code, but like it's better for PyTorch to not actually have a required dependency on Numpy.
So if we don't even wanna depend on Numpy, well, we certainly don't want to depend on f f m peg and domain libraries also let us do it this way.
Another reason why domain libraries exist is they actually have different contribution models than Pytorch's main repository.
If you've ever submitted a pull request to Pytorch, Pytorch, you may notice that after, you know, code review and all that regular stuff, someone actually has to go ahead and import that diff into fabricator.
That's Facebook's internal CI system.
And then only then, like, there's some complicated land process that, you know, if you're external to Facebook, you're not really privy to, but eventually, maybe a week or two weeks later, your PR gets merged.
Oof.
That takes a really long time.
Hopefully, like, it's not too bad.
One of the things that I worked on a lot isn't making it easy for open source people to work on Pytorch.
But yeah, that can be quite a bit of a lift.
Unlike Pytorch, Pytorch, all the domain libraries don't directly sync with Facebook, so we actually have many external contributors who have direct commit access to these repositories, you can land stuff a lot easier and sort of there's a sort of calculation we're doing here, which is that Why does Pytorch Pytorch, you know, insist on every, you know, commit you land also immediately going out to Facebook production when you land it? Well, it's because Pytros has a lot of moving parts.
There are a lot of systems that depend on it, and so we can help de risk this by like continuously deploying our changes and like just seeing as soon as possible when things break because there's a lot of moving parts, there's a lot of interactions It's better to, you know, learn about them early rather than, you know, in a weekly release where, like, oh my god, there's so much stuff and we broke everything and no one has any idea what broke what.
But domain libraries, one have less applicability.
Right? Like you're not going to use a text library if you're doing a vision processing task, unless you're, like, you know, doing labeling or something like that.
deep learning is all sorts of, you know, interesting cross pollination.
But and furthermore, there's way less code in there.
Right? Like, it's mostly stuff that is specific for the domain question.
And so it's not so bad to just periodically sync in this case.
It it can be a little bit troublesome, but it's less bad.
And so, you know, sort of, these repositories live on separate ends of the scale.
So, yeah, if you wanna move fast and you want to, like, be able to, like, you know, sort of work on things very rapidly, it's usually a lot easier to do that inside domain library.
than outside.
Okay.
So that's a very developer specific viewpoint on domain libraries.
And the next question I wanna answer is What does a domain library do? Right? Like so when I talked about what Pytorch is a project, well, what do we do? We give CUDA accelerated operations that have automatic differentiation and, you know, a bunch of, like, extra stuff to make it possible to do stuff around it like, distributed and stuff like that.
So domain libraries are really very similar to many of the things that we do in Pytters core.
Right? So one of the bread and butter things for a domain library is it implements operators like ROI align that don't make sense in a general context, but are very useful in the, you know, context of the domain question.
Actually, in the old days, even Torch Vision used to be a pure Python project.
So actually, these operator implementations would just be comp compositions of stuff you found in Pyturgical Core.
But as having on, you know, there there's a need to have accelerated kernels.
And so, torch vision and most of the other domain libraries are proper c plus plus libraries and they come with actual optimized operator implementations for the for various situations.
And these are also done with autograd support because obviously you wanna train your models And, yes, we provide CUDA kernels because GPU acceleration is a really important thing of what, you know, makes deep learning tick today.
So that's very normal, but there's also some operators that you'll find in a library like church vision that are unusual, like not sort of what you'd expect to see.
So for example, one of the things you need to do a lot in domain is you need to be able to encode and decode the file formats for your domain, like, you know, the JPEG example I gave earlier.
And as I said, you know, what a domain library is doing for you is it's getting all the dependencies.
So most of our domain libraries don't actually implement the nuts and bolts of encoding and decoding because there are plenty of good open source libraries for doing this.
But what, you know, the domain library is gonna do is gonna take care of getting the dependencies for you either, you know, like because there's some other condo package that does it for you or or maybe it's some library that's very annoying like socks.
And, you know, you like, if you had to install it yourself on Windows, that would be really annoying.
But fortunately, for you, Torch Audio actually just bundles it with the biners and questions.
So you just can use torch audio directly and get those implementations.
And, you know, sometimes we even, like, create custom objects for representing various concepts in them.
There's a API in PyTorch called Torchvine for representing these things.
And so, you know, it's both a data model as well as operations for working on them.
There are a bunch of other things though beyond operators that a domain library does.
So for example, domain libraries often come with models, and it's especially important they come with pre trained weights.
pre trained weights are wonderful.
Right? Because not everyone can be Google and have a bazillion, you know, GPUs to, like, train your model.
Well, yeah.
So, you know, pre trained weights let you you know, if you don't have that many GPUs, you can, like, use something that someone trained on a big data set and then, like, go and fine tune or, like, you know, try to put things together that way.
So, you know, envision, there are plenty of vision models like the good old fashioned resnet but then a lot of more modern models.
And, you know, towards vision, the intention is to actually track, you know, the models as things go on and just be a one stop shop like okay, you're a researcher.
You need a, you know, reference implementation because you want to compare against some baseline.
Cool torch visions got you covered.
or, you know, maybe you wanna take some model and then tweak it, well, you can also look in towards vision and get the model that way.
Similar to models is datasets.
Right? We talk a lot about in deep learning how, you know, models are the stuff we're training, but a model is only really as good as the data you feed it.
And there are a ton of, you know, well known data sets that, you know, are done for various tasks.
And Tor vision makes it easy for you to, like, get all those data sets in a, you know, uniform API, then feed them to data loader, which, you know, you can use to kick off the rest of your Pytage program.
And, you know, like, they even have reference scripts, right, to, like, show you how to do the end to end training.
You actually need to establish a baseline or you wanna do some sort of ablation study or something like that.
There's a few other things that like are not as obvious.
So one is that, as I said, domain libraries often need various dependencies, and they take care of making sure all these dependencies are available for you.
And one of the important things that, you know, makes this possible is we actually do distribute binary packages for domain libraries.
Right? Like, this is this is probably, like, one of the hardest things about, like, running a domain library is when release runs around and you need to build binaries and, like, building binaries is very complicated because you need to do it on all the and you need to get all the dependencies.
You make sure they're linked correctly and stuff like that.
And so working inside of domain library, that is one of the things that they do for you.
It's also one of the reasons why It's a little it's a little hard.
We we we've been stuck at three domain libraries plus a few experimental ones for a while because it is a lot of bring up to get all the packaging going.
But it's one of the value ads of, you know, working inside one of these domain libraries.
And finally, and this used to not be true, but it is increasingly becoming more true is R domain libraries are compatible with deploying to mobile.
At some point, I should do a podcast about, you know, what's going on with mobile and Pytorch, but like suffice to say that, you know, you can take your PydISH pod models and run them on the phone and we are doing this at Facebook.
And domain library is, right, while they contain stuff for doing images and audio.
Well, those are very much the types of things you'd like to do on your phone.
And so actually, no torch vision is compatible with Ashley running on the phone despite being in a separate repository.
You know, that's kinda ridiculous and I don't have time to talk about how that all works.
But it's pretty cool and it's another one of the things that a domain library does for you.
So I've talked a lot about, you know, why the domain libraries exist and what they do for you.
And I wanna go back and reexamine this question, which is, well, you know, it sounds great to have the domain libraries being these separate modules that are external from Pytorch, Pytorch.
What did we give up when we did this? And in particular, the thing we gave up is that these libraries have to be loosely coupled with Pytorch.
This should be a familiar conundrum to anyone who has ever had to deal with a system where you were due you had multiple components that had different release cycles.
Right? Like, if you are in this situation, you're not in a mono repo where everyone is running off of the latest version of everything all the time, well, you know, you can't just land a change to some base library like Pytorch Pytorch and then expect to be immediately able to use it in your library.
Right? The base library has go ahead and do a release, and then you have to go and update your stuff to actually use it.
That being said, Plyarch is not very ABI compatible.
So we whenever we do a new version of release of Pyarch, we always do a new releases of all the domain libraries as well.
So we we do have some level of coupling.
Right? Like, so if you're looking at, like, torch Visions, CI, we are it actually runs against Pytorch Nightlies.
Right? And because the APIs that the domain library use don't change that much.
Most of the time, this is working.
And actually, the PyTorch main CI itself also actually test against TorchVish So and one of the CI jobs, we will go and we will build torch vision from scratch.
Remember, torch vision is not that big of a library.
It only does stuff for vision.
It's not like you know, gargantuan library like Pytorch.
So it does take that long to compile and then we can quickly test and make sure that stuff works.
But there are some APIs in Pytorch which sort of move a lot.
And we change them a lot like tensor iterator.
And, you know, it would actually be kind of useful to be able to use these tools in domain libraries, but then stuff will break all the time.
So people just don't do that.
They only work on the stable APIs.
This would be kind of nice to like make some improvements on.
like, maybe sometimes you might want to, you know, write a new binary operation that's very specific for a vision.
But today, mostly, if you need something like that, you're just gonna go land it in Pytorch itself.
And so, you know, it's just a little hard to coordinate changes across multiple repositories.
So people will, people generally have a volatile code to not require this in this way.
I'm almost done with talking about domain libraries.
One last thing I wanna say is that, you know, when you're working on domain libraries, the users matter a lot.
I'm so I'm here, I'm talking to, you know, you you developers who, you know, like like writing code and don't know that much about machine learning.
Right? So when you're working on domain libraries, right, you're very close to the actual research that's going on that domain.
Right? because I talked about how, like, the libraries provide models, they provide datasets, and and, like so you need to actually be keeping track of what's going on on the research side.
A really good example of this is Francesco Masa, the main maintainer of torture vision.
Francesco does a wonderful job taking care of George Vision.
And he also does research on the side.
Or or maybe half like, you know, it's one side is treasures and the other side is research.
There are a lot of really cool papers that Francesca has been a part of and, you know, this is, like, I I think of this as one of the, like, big reasons why George Vision is so successful is that we have someone at the helm who, you know, knows a lot about implementing framework stuff, but also knows a lot about the research stuff.
me, I'm, you know, always in core, like, you know, c plus plus, you know, core abstractions land.
And I actually don't have to train models very often.
in my job function.
But, you know, in domains, you gotta be doing that sort of thing.
That's everything I wanted to say for today.
Talk to you next time.
.
