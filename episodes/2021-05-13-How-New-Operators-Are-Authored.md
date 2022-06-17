---
layout: post
title: "How New Operators Are Authored"
date: 2021-05-13
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# How New Operators Are Authored

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I'd like to give a short intro slash primer about the general developer experience that happens when you wanna add a new operator to Pytorch.
Despite, you know, Pytorch being a library for doing numeric computing and so you know, hey, you know, what are we up? What are we all about? Well, we're all about a big pile of operator implementations for all the things you might wanna do.
Actually, it's not that common that we go about and add a new operator Pytorch.
It's actually pretty rare because we kind of have a lot of operators in Pytorch and, you know, most of the time, when you want to do something interesting and new, usually, you just, you know, put a bunch of operators together to do whatever it is that you are interested and doing.
And that's like, you know, that's basically what people are doing when they write deep learning models.
Right? They're just putting operators together into bigger and better operations.
So you only really need to write a custom operator when there is something that you need to do that like sort of can't be done efficiently by putting everything separately.
So, like, kind of classic example, which applies to PyTorch and is sort of ameliorated if you've got a fusion compiler, is if you say wanna write a new point wise op that consists of a bunch of point wise operations and you don't actually want to, you know, run them separately one by one, loop by loop.
Right? Because that takes a lot of memory traffic.
Well, then writing an operator for that case is quite a big benefit because once you fuse them together, things can run substantially fast.
But okay.
So let's say that you know you're actually doing some sort of really fancy linear algebra or you need a new point wise fused up.
or, you know, any sort of situation where you, you know, need the performance that you can only get from running a kernel.
What does it look like when you wanna add a new operator? Well, there's sort of two main modes that people write new operators in Pytorch.
One is adding a new operator to the library proper.
So this is, you know, this is core Pytorch the next release of PyTorch, the operator is gonna show up, and, you know, you can make use of it.
It, you know, is something that you put in native functions dot YAML.
A file will be talking about more later in this podcast, and it's just something that we consider in core.
But there's another way to write a new operator in PyTorch, and that's using the operator extension mechanism.
So using the torch library header and macro, you can actually define operators completely externally from Pytorch And then you can just, you know, you register them via a PiBE and eleven like registration system.
And then these operators become available for you to use by the torch dot ops name space.
So, you know, there's there's a difference between these two things.
Right? If you add a new operator to core Pytorch, The thing you need to do is you need to make sure everything works.
Right? So you need a CPU implementation, you need a CUDA implementation, you need working derivatives for it, you need, you know, comprehensive tests, like auto grad ground checking, all that stuff.
And sometimes oh, and not only that, but, you know, your operator needs to handle all of the different kinds of tensors that, you know, a PyTorch program might throw at you, including tensors with strange strides or different layouts or very different d types.
Now if you're just someone you know who like just needs a little code that works on floating points just for this particular case on CPU, often you don't actually wanna go through all that rigmarole.
And also, maybe your operator is, like, not very well defined.
Right? It's not doesn't mathematically make sense.
It's not really something of general use.
It's just something very specific for your problem.
Well, writing a custom operator is great for this use case.
Right? because you just write out your operator and you do the thing for exactly the use case you need and no one else is really bothered by the fact that you wrote a custom operator like this.
So the the use cases in these two situations are kind of different.
But let's talk a little bit about what happens when you add a core operator to PyTorch.
So what exactly does this entail? So the first thing you need to do is you need to define what the API for this operation is gonna be.
And the reason for this is Pytorch is not just a Python library It is actually, you know, also a c plus plus library that you can use directly from c plus plus.
And it's also a compiler and interpreter that, you know, you can interpret Pytrich programs on.
And so, you know, you don't just write an operator by writing a new Python signature.
We need to write a API declaration for the operator which is generic across all of the different modes of use, interpreter, c plus plus, Python and other situations that can work in all those cases.
And what we call this, you know, specification is a Jitskema string.
So if you're in Pytorch core itself, there's this file called native functions dot yamal.
And what it has is it has all of the Jit's scheme of strings for all of the operations that you might be interested in.
And just schemas strings are like some sort of mash up of the python type system and the c plus plus type system.
So, you know, you can say, okay.
Well, my first argument is a tensor.
My second argument is maybe a integer list because I need to, like, provide what the padding is.
The schema also knows about aliasing.
So, like, what if I have a function does the input alias with the output.
And it also knows about, like, mutation.
Like, is my function purely functional? which is most functions in Pytorch, or does it, you know, mutate one of its inputs, and you have to tell it that too.
You don't need this information if you're just writing Python code.
But you do need this information to say if you're a compiler and you're trying to understand whether or not it's safe to, you know, do a code movement optimization or not.
Okay.
So that's cool.
So you write this entry and native functions dot yamal, and what this does is it triggers off a very long sequence of cogeneration pipeline, which actually goes ahead and generates python binding through your program, the c plus plus binding through your code, etcetera, etcetera.
And so all you need to do after you define one of these native functions dot yamal entries, is you just need to provide an actual CPU and CUDA kernel.
And so, you know, in the Yamal file, it's not it's not I'm not really here to, like, tell you exactly how to do this.
If you wanna like look at the actual code, you should look at some of the further reading links after this podcast.
But what, you know, your what you're gonna do, right, is you're gonna write CPU implementations.
You can say, okay, my CPU invitation is gonna be, say, softmax underscore CPU.
And one of the things the cogen does is it generates a header stub.
And what this header stub says is, hey, here is the c plus plus function spec you to have written.
And once you write it, then I will, you know, do all the necessary plumbing to make it possible.
to, you know, call into your kernel.
Now the way you do this is a little different depending on if the kernel is structured or not.
See you're at my previous podcast about structured kernels.
But the same con the the general concept is the same.
It's just we generate slightly different stubs in the two situations.
there's different code you have to write.
You have to write more code if you're doing the old fashioned way because you have to also define the out variant and the in place variant directly and the search your kernel version takes care of all of that for you.
But as a result, you have to, like, structure your kernel a little differently.
But it's it's very very similar.
Okay.
So you've got to this point and you've got all the scaffolding that you need to actually call your operator.
How do you actually implement your operator? Well, as I said, in PyTorch, we expect you whenever you add a new operator to give a CPU and CUDA implementation.
So what does a CPU implementation typically look like? Well, normally, if you're doing some CPU code, it depends on how complicated it is, but pretty common situations are, for example, there's some external library that's already written efficient CPU kernels, and you just go ahead and use those directly.
Right? So in that case, all you're doing in the kernel is, you know, you got some tensors, you figure out, you know, how to what their data pointers are.
You make sure that, you know, all the invariance that the library expects are upheld, like, that the inputs are contiguous.
Most libraries don't handle discontinuous inputs.
It's pretty uncommon.
And then, you know, you just call under the function in question.
And maybe you have to go and allocate the output sensor for it to write into.
But if you're actually writing a operation yourself.
Well, there's a few facilities for writing very common styles of operations.
In particular, if you're doing a point wise operation or auction.
We have this really useful class called Tensor iterator, which takes care of all the sort of gnarly details of, you know, like, if I have a tensor in a different layout, how do I, you know, restride it so that I iterate over all the different strides without, you know, processing memory that's not necessary.
Blah blah blah blah, do all of those things and then all you have to write is a little lambda That says how to actually do the point wise operation in question.
So, you know, and then all the others infrastructure taken care of for you.
This only works if you're doing one of these very simple, you know, like, point y cell operations or there's a few other cases like, you know, a refrigerator can also handle reductions in some sense.
If you don't have a CPU kernel that falls into this these categories, then you might actually have to let you know, oh, goodness me.
Right? some efficient CPU code that actually does the thing you want.
Sometimes, you know, it's simple it's easy enough to just write a plain old four loop and c plus plus because Maybe you don't need it to be that fast.
It's just that doing a four loop in Python is too slow.
And there's also like lots of other libraries that try to build off of this.
Right? Like, a number and Python.
All these ideas are like, oh, yeah.
You know, like, Python's really slow, but, like, maybe you wanna write numeric loops in Python and then they compile a sequel plus.
Well, it's not too hard to write these loops and c plus plus as well.
And so in PyTorch, people usually do that.
We provide a bunch of facilities for, you know, doing common optimization For example, if your algorithm is parallelizable, you can use the parallel for loop construct that we provide to, you know, farm out your computation onto different threads so that you know you can take advantage of multi threading.
And, you know, if your kernel is running slow, well, typically, colons are pretty short.
So you can, like, easily run it under perf and then take a look and see, okay, you know, am I missing cache a lot? Am I spending a lot of time on instructions, you just do normal techniques for optimizing performance in this case.
Optimizing numeric code is is different.
Right? Like, people always like like to say, oh, yeah, you know, matrix multiply, how do you implement that on CPU? while, you know, you really need to know about cache.
And so it's very different from optimizing other types of code.
But there's also a sense in which optimizing CPU code is very easy because Sorry.
Optimizing CPU kernels is very easy because there's not just not that much code.
So you can actually come up with a pretty good mental model of what you're supposed to do in this situation.
Okay.
So that's it for CPU.
What about a CUDA kernel? Well, CUDA kernels are pretty similar.
Right? Like, we need to do all the same things except instead of writing CPU code, There's this CUDA programming model and we need to know things about how the device actually works.
But then you're still writing CUDA and many of the things that, you know, you like expect to see in CPU, you know, the CUDA ecosystem is well developed enough so that alternatives to these things also exist in those situations.
So for example, if you need to debug your CUDA kernel is crashing while there's a tool called CUDA MEMCheck which will tell you about, you know, what the what is causing crash.
You can also In a pinch, use CUDA GDV, which actually lets you step in problem.
You can also add a search to Colonel's good old fashioned print f debugging.
And, you know, if all else fails, well, you know, once again, your pseudo kernels are usually pretty small.
So you can, like, maybe, bisect your way to figure out what the error is.
really the hard part about writing a CUDA kernel is actually understanding the device model enough so that you can actually write conquering code.
And so if you ever, like, look at a presentation about how to write CUDA programming, like, what they're actually gonna do is they're gonna spend a lot of time talking about how these processes actually work, you know, what the, like, actual physical details of the hardware are.
Because this actually really matters if you wanna write efficient code.
Of course, if you're doing something simple, like a point wise op, while it turns out sensor iterator also works in that situation.
So you can just, you know, use our our, you know, scaffolding in that case.
but it's actually kind of challenging to write a good CUDA kernel.
And an example that I'm thinking of recently is we were working on some linear algebra code and the algorithm that, like so a lot of the times, right, there will be a well known CPU implementation, and we wanna add it to PyTorch, and we need to somehow figure out how to GPU accelerate it.
And so this CPU implementation in question had a problem which is that it needed to do a little bit of computation at first to figure out how many iterations of approximation it was gonna do.
Well, basically, we were doing, like, these tailor expansions for the computation in question.
And we needed to, like, look at the conditioning of the matrix to figure out, like, how many taylor expansions we needed to do.
And so I remember reviewing this, the CUDA implementation for this PR and us arguing about, well, you know, we can't actually, on CUDA, make a decision based on the data what to compute on without doing the synchronization.
Because remember, CUDA's async.
And so if we need to, like, look at the data and question, we have to wait for whatever prior kernels we're running to finish running, to give us the data.
And then we need to run our actually operation and then get it to CPU.
So we, like, talked over and, like, you know, benchmark a bunch of different options.
and it turned out it was still it was still better to synchronize so that we could pick a good tailor approximation for this case.
But, like, there's gonna be a lot of problems like this where, like, you know, it's not easy to program a GPU.
And so you're gonna have to actually understand, like, there there's actually, like, non trivial technical content and, like, recasting an algorithm so it works on CUDA.
But let's say you do that.
Right? So you've got your CPU kernel and then you got your CUDA kernel.
and you've plugged it all in via the native functions dot yamal system.
Well, then you're basically done.
That's it.
You've got some more stuff to do.
Right? You gotta test your operator.
and we have a bunch of facilities for testing in Pytorch.
But they all involve, you know, just like running the kernel in question and, you know, well, you've already got bindings provided for you, so it's pretty easy to get that hooked up.
We have a bunch of stuff like, for example, Mike Wooberry's been working on a new op info extraction.
which lets you describe some properties about an operator and then we can automatically run tests based on what kind of properties the operators on hold.
Unfortunately, these kind of mostly are for, like, urinary and binary ops.
They're very simple types.
They're very regular, and there are, you know, simple things we can check.
But, you know, there are there are also some, like, very generalizable checks we do.
For example, there's a check-in our test suite called GRADCheck.
What does GRADCheck do.
Well, remember that when you're writing an operator in patterns, you also have to say how to differentiate it.
So we typically have symbolic derivatives for all of our operations.
Usually, cast in terms of other, you know, functions that you might have to implement.
Well, what GRADCheck will do is GRADCheck will use your analytic.
Sorry.
Not analytic, not symbolic.
It'll use your analytic derivative formula and it'll also numerically compute what the derivative is based on your forward implementation.
And then it'll just compare the two and figure out whether or not, you know, they agree or not.
And then they don't agree, GRADCheck will fail.
and this will work for any differential function you have.
You don't have to write a separate test for each of them.
But yeah, so you add some tests and, you know, you have to write your docs for the new operator and, you know, you've got your colonels, and then that's great.
And usually, you, like, submit the PR and you give some benchmarks.
Like, it's very easy to benchmark kernels when skin because they're, like, very regular and you can just try them a bunch of different different impensizes.
And then you're off for the races.
Really, the hardest part is convincing Pytorch that we actually do wanna take your operator.
That's a story for another time.
That's all I wanted to say today.
Talk to you next time.
.
