---
layout: post
title: "All About Nvidia Gpus"
date: 2021-09-20
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# All About Nvidia Gpus

Hello, everyone, and welcome to the Pritchard Dib podcast.
Today, I have a special guest with me today, Natalya Kimelshin, who's gonna be here to talk us about all the various CPU architectures.
Natalia, do you want to introduce yourself real quick before we start? Hi.
I am Natalia Gomoshuan, and I am considered a GPU expert around Facebook perhaps undeservedly.
But anyway, GPUs are going to be the subject of our podcast today.
Alright.
So when I was thinking about, like, topics that I wanted to bring in other folks to talk about, one of the things was sort of just hey, there's a lot of, like, different Nvidia cards out there that do all sorts of different things and, you know, sometimes when you're new, you hear things like a one hundreds and v one hundreds and how they have different performance characteristics.
And I just wanted to, like, talk about this a bit and, like, you know, sort of get, you know, a sense about, like, what's actually important? Because if you just, like, pull up, say, the Wikipedia page that says about all of the devices that Nvidia has, There's a ton and ton of different cards.
Which ones matter? Which ones don't? How do I actually understand them? You know? So that's that's what I kinda wanna dig into today.
with Natalia.
So I guess we should first start off by talking about GPU architecture.
So so Natalia, what exactly is a GPU architecture? At least in NVIDIA's terms.
In NVIDIA's terms, GPU architecture has pretty much the same meaning to it as a CPU architecture.
is just a number of capabilities that this particular generation of the GPUs has.
How fast can it process floating point numbers, how fast can it process low precision numbers, how fast can it do indexing computations? how fast can it read and write memory? How many register and shared memory does it have? So all those characteristics constitute GPU architecture.
And each next one, of course, is considered to be better than the previous one.
So so what's an example of one of these GPU architectures that we're talking about? The most recent GPU architecture is ampere.
So those A100 cards or if we are in consumer loans, then thirty something cards.
that is the latest architecture that both obviously the best performance known so far.
The previous architecture would be Volta and Turing cards that are still excellent cards and still used a lot around many places.
And we can go back to the previous generations, but I guess we'll do it a bit later and in efforts order and not in reverse.
What's the difference between Volta and Turing? The difference between Volta and Turing is that Volta is mostly a data center card and it is the first card to introduce the chance of course.
That is the thing that allows NVIDIA to do very fast low precision matrix multiplications.
During his kind of consumer brother of this server card and an additional capability that Jira has is fast integer computations that can be used for very fast contest in France.
Okay.
So we've so far talked about three GPU architectures that NVIDIA has released.
Ampere, Volta, and Turing.
So ampere is the latest and greatest.
And you've mentioned a bunch of different things that, you know, distinguish these cervistic.
So let's talk about AMPYER and Volta for now staying in the data center.
So like, what are the big differences between these architectures? So the big difference between AMPYRE and Volta, the one that's probably most important for us, is that AMPYRE has introduced couple new data types for our merchant's computations.
One is, before sixteen, the data type that has been used for a long time on GPUs.
That's also occupies sixteen bits in memory, but doesn't suffer from the same problem as the older low price decision NVIDIA type FP sixteen suffered from.
Because FP sixteen has a very small dynamic range and is prone to underflooring and overflooring.
So a lot of numerical tricks have to be applied to avoid this.
Before sixteen, his fear meant is a bit but as many participants as regular FP32 type.
So if you are not over under flowing with FP32, then before sixteen will probably provide you with more stable numerical characteristics.
And the second type that I mentioned, TF32, TensorFlow thirty to do.
Is a weird thing that's aimed at deal practitioners who can get split ups right out of the box.
So NVIDIA's claim here is that if you are using t f thirty two, then you don't really have to do anything to your existing FPS thirty two program.
If it works with FPS thirty two, it's supposed to work with TF thirty two except it will be much faster.
And the reason it will be faster is that when GPU is doing matrix multiplication instead of reading all your anticipates a bit it will read just a few of them and perform lower precision matrix multiplication that will be much faster, but he will still be left with your thirty two bit container.
He will still be left with all your dynamic range.
And Generally, you'll get the same results faster.
So if I don't use any of these new features, and I upgrade from a v one hundred volta to a a one hundred ampere.
Do I expect my code to run faster? Yes.
You would expect your code to run faster because the peak performance for A100 is noticeably better than big performance of V100.
And that is both for bandwidth, a band called memory bandwidth for a one hundred is higher and for compute bond codes because peak compute performance for a one hundred is always higher.
And you should be using at least one of the low precision data types if you're running on v one hundred and a one hundred because that is their main claim to fame.
If you are just doing your plain f p thirty two computations, you are throwing away a lot of computer power that's v one hundred and e one hundred LRU.
Alright.
So that's cool.
So here's a question for those of us who don't work at Facebook.
If I wanted to play around with an a one hundred or b one hundred, is there any way I could actually you know, get my hands these cards without having to buy it? So one thing is you generally should not be buying data center cards.
because they don't have active calling.
You cannot put them in your desktop rig even if it's a very good rig.
So don't buy them, please.
You won't have any use for them at home.
But if you want to play around with them then AWS has instances for both v one hundred and a one hundred.
They are not the cheapest but probably you can find someone who would help you with this.
Or if you want to just play with the consumer equivalent of those cards, then yes, you could buy a thirty series of the GPUs.
hopefully, you can buy them by now because a few months ago, it was a big quest.
They got sold out at the moment.
They appeared and it was incredibly hard to buy them.
And unfortunately, I don't know the exact situation on the ground now, but hopefully it's better.
Just to confirm so, So what we've been talking about, the a one hundred and v one hundreds, those are the data center GPUs.
But like when I talk to, like, gamers, or like, you know, machine learning enthusiasts who just have a few GPUs in their basement, they're going to be buying different things.
They'll they'll still be ampere and Volta.
Is that right? They'll still be ampere and tearing.
Volta, didn't really have a consumer credit card.
They had something, but it's hard to find and not necessary.
Okay.
So I wanna change the topic a little.
So when I look at the Pyturbine code base, typically, I don't see any references to AMPYRA and Volta.
specifically, except maybe in comments here or there.
Instead, I see a lot of references to SMs.
So, like, for example, when we build Pytorch, we can specify what set of architectures we wanna build for via like the torch architecture list.
And usually I have to list a bunch of like these SM fifty one blah blah blah, things like that.
SM is the core part of a code architecture.
So GPU is a massively parallel processor.
And to enable this parallelism, each GPU consists of a few ASMs.
For the first GPU generations, the number of ASMs was on the order of ten, say, ten to twenty.
For the recent generations that we were talking about, it's closer to one hundred SMs.
And each SM in turn is handling about two thousand threads.
So you can see the level of parallel execution that's going on the GPU.
And SM pretty much has everything that is needed for the GPU to process the data very fast.
it has a few compute cores that would be doing your integer well precision of floating point computation.
It has on chip memory that can be used to very quickly access and write some intermediate results.
And of course, it has the schedularies that would tell the threads when it's time to go execute something and when it's time to wait.
So if I actually wanna talk about the nuts and bolts about, like, you know, what actually we're targeting, I don't talk about AMPYRA or Volta, just talk about, you know, what architecture these actual SM units support in the in the chip.
Is that right? Yeah.
But there is also more or less one to one weapon between those sixty one or seventy or seventy five architectures that you specify when you are comparing the code and more human readable vault during AMPYER.
that we are talking about here.
What's your favorite way to remember what the correspondence here? I don't unfortunately have a a favorite way.
It's just, you know, when you see it enough times, then you remember.
or look it up on Google, hi, so far.
Yeah.
And on the Vicky pages, there is also usually something.
So there's one more piece of the puzzle.
In talking about the hardware, like the actual, you know, silicon you get for any of these GPUs, But there's also another part which is important, which is the CUDA version that, you know, you're using to actually, you know, run the software stack on top of these GPUs.
How should I think about the relationship between CUDA versions and the various GPU hardware that I might be using? Actually, it's not just the version, there are two pieces of software that are required for you to use your GPUs for computation.
One is CUDA capable driver that comes with its own version system.
And another one is CUDA talk it.
which probably you referred to as CUDA version.
And so all these three components that are hardware driver version and CUDA version have to be in sync for you to be able to use your GPU.
Exactly in sync? No, not exactly in sync.
And this relationship have been relaxed recently.
So it's even easier to get to the working configuration than it used to be, say, a year or two ago.
But generally, if you have a card of some architecture, let's say, ampere, there is the minimum version of the CUDA toolkit that you need to be able to compile a code that will run on this architecture.
and for ampere a one hundred architecture that would be Coudé eleven.
If you bought consumer grade card thirty series that would be eighty six architecture and that would be quarter eleven to release.
then your driver also should be at least the necessary version to run this hardware and four ampere cards that again would be a driver corresponding to eleven or eleven dot two tool kit.
Now I was saying that it doesn't have to be exactly in sync.
Previously, your driver version had to be newer or the same as your tool kit.
Now with enhanced driver compatibility, NVIDIA allows the older version of driver for the newer toolkits just as long as the major version for both driver and tool kit match.
There is also yet another way you can run your newer cards with older software and that will be you still need the driver to be able to run the code.
But if the tool kit that you use to compile is too old and doesn't support the newer hardware yet, you can still compile not to binary but to an intermediate thing called PTX, and then this PDX would be compiled to a binary by the driver itself.
And even if you have some pretty old codes compiled with the old CUDA tool kit, you could rely on the driver to just compile it and be able to run it within your cart.
This is, however, not recommended because a this jit compilation will be pretty slow.
If you want to, for example, run.
By rush in this mode, you will have to wait for half an hour to an hour for all criminals.
to be compiled.
And then the performance will probably be pretty bad.
So you will be able to run, but you won't be happy that it is.
half an hour, that's really long.
Yes.
That's really long.
And we're actually having big discussions, whether we should disable this thing and error out altogether or whether we should allow people to do it And there are some companies that that have a hard time distributing new versions of software that actually rely on this being able to run all software on the newer cards.
So that's why we cannot disable it outright.
Even though for In in most cases, I think people would be happier just airing out and not seeing how long it takes to do something.
Alright.
So changing topics again.
So we talked a bunch about a one hundreds and v one hundreds, the data center versions of the cards because We work with, you know, a bunch of people who are running their deep learning models on big research clusters that like, you know, have lots of GPUs of this kind.
but GPU usage outside in the wild is very wide and heterogeneous.
Are there other models that are worth knowing about? in this market.
We already talked about the consumer grade thirty series.
Anything else people should know about? Well, if they cannot get a thirty series for some reason or if they want something cheaper than During carts, there is seventy five series, are still an excellent thing, and they are good for gaming.
not as good as Thursday series and videos.
But still, they were – the first one, introduced ray tracing technology.
And they still provide a pretty good performance for the computer workloads.
I guess if someone wants to use GPU for their small project the biggest consideration is probably the amount of memory that the particular GPU has because in most cases here workloads would be limited by how much you can put on your GPU.
Just get some recent video charactering or and peer series make sure that it has, I don't know, eighty gigabytes memory at least and at least for some small experiments that should be enough.
I heard Google Colab will also give you a free GPU on what what model is that? They used to give you free Turing GPUs.
But recently, all my collaborations that I was able to get were just K-80s, which is a Kepler GPU that's pretty old, wasn't produced to in in in twenty fourteen, if I'm not mistaken.
Wow.
That's really old.
Yeah.
That's really old.
But I guess as PyTRGE developers, that means we do have to ship Kepler compatibility by default.
Don't we? Yes.
I guess since both Colab and AWS still have those key instances we do have to support them.
Okay.
So that's everything that I had on my topic list for today.
Natalia, are there any final closing thoughts you wanna give us before we close-up? I do appreciate that we talked a lot about consumer grade cards because this is what most beginning researchers are working with.
And that's their introductory to CUDA.
And I'm very happy that when NVIDIA started CUDA, they made this decision that every absolutely every CUDA, every GPU is going to support CUDA, not only like higher level of models, but pretty much everything.
And fun fact of the first ImageNet competition that was that Alex Krizovsky one with his Alex Netz.
It was trained on a couple of consumer grade cards, so it shows you that consumer grade is all in it, basically.
That's pretty cool.
Alright.
Well, that's all we had to say for today.
Talk to you all next time.
.
