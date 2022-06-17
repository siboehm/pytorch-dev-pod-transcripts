---
layout: post
title: "Expect Tests"
date: 2021-06-16
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Expect Tests

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want about talk about expect tests.
Expect tests are a form of test that have a characteristic property, which is that when the test fails, you can automatically ask the test framework to update the test for you to accept the new output.
And so sometimes they're called golden tests because there's some golden version of the output and you can refresh the golden version based on directly what the test tells you.
So imagine that you've got some program and under some cases it raises an error.
You might not want to hard code the error as a string to compare against because well what if someone edits the error then you'll have to go and find all the places and update, you know, the error message that has been hard coded in those places.
And often what people will do is they'll write a regular expression instead in this case to match for some singular important piece.
But, you know, if you completely rewrite the error message, that's no good either because the reg x will probably fail and then you've got to go and go to each of the sites and manually fix them all.
So what expect tests to say is that, hey, When this happens, when your test fails, you just rerun using expect test except equals one as an environment variable or something like that.
and then the expect test will automatically go through all of the sites that were wrong and modify them so that they have the new output in question.
And then for example, you can just run git diff to go look at the changes that were applied and see if you like the changes or not.
it makes it really easy to write tests that track internal implementations details really closely, while still making it not so painful to update the tests as those implementation details change.
Expect test long pre date Pytorch and this podcast.
My personal story with expect test is I first ran into them while working on GHC, the Haskell compiler.
GHC had expect tests and the way they were used was to test the error messages that GHC gave because, you know, one of the things that Pascal is very famous for is a very strong type system.
And so we work very hard when you have a type error to give you some useful information in this case.
And, you know, there are tons and tons of test cases, testing what happens when things are mistyped.
And, you know, when Simon Pin and Joe's goes and rewrites, you know, exactly how unification works or, you know, some other major subsystem in the type system, chances are a lot of error messages are gonna wobble.
And having accept tests meant that it was really easy to just go ahead and update all the error messages and then just like eyeball them and see if they made sense or not.
I actually, during my PhD, ported a version of this mechanism over to Kabbal's test suite where, you know, cabal also, you know, has a test suite where you wanna run lots of different cabal packages and see if they compile or don't compile.
And it was kind of a pain to, you know, write exactly what you expected it to happen and accept us made this easier to do.
I'm also reminded of a conversation that I had with Ron Minsky at Jain Street where he was describing to me some of the stuff they were doing with accept accept tests in Jain Street.
And their model was that unlike what GGC was doing, which was we had these files that were the expected values.
And so when we refreshed the test, we just updated those files.
What they were doing was they were actually storing the expected strings inside the test files directly.
there are a lot of good reasons to do this.
Imagine you're writing a test case.
Right? So you like set up some functions, you do some operations, and then you wanna like check that the output makes sense.
It's much easier for a code reader and a code reviewer to understand what the test is doing overall.
If the actual expected value of, you know, whatever string comparison is directly in the test file.
Right? Because you just read what the test setup is, and then you read what the output is.
Now, this is a little challenging if you want to, you know, do the main property of expect tests.
which is that you can actually update the test output automatically.
So what you basically have to do is you have to write some code that knows how to update you know, your Ocampo code or in Pytruition's case, your Python code, and rewrite the source code so that you actually can put in a new value in question.
But if you can get past your distaste for doing this kind of thing, it's really really helpful and makes except tests a lot easier to understand.
One of the sort of complaints with, you know, expect tests that, you know, are in an extra file is they're just these random files.
And when, you know, you have to update these test files, it's basically you got this big directory of all these expected things, and they're all wobbling.
And you have no idea why they're wobbling or not.
Actually, a lot of On X tests when we originally wrote them were done in this way because it was easy to do it this way.
And I don't know.
I I think they're not very interpretable, and people decided they didn't like it very much.
But you know what the answer is, just put the expect test directly in your source code.
There's another important function to doing it this way because you when you were putting it directly in your source code, you don't want your expect test to be too long.
And figuring out exactly what you are going to test against, what you're actually going to record in your test file.
That's pretty important.
Right? Because you don't want it to be too long, but you don't wanna be too short to not capture the important things you wanna do.
So actually, when you are going in and writing an expect test, you need to think about what exactly it is you want to output in this situation.
And so when I'm bringing up a new subsystem and I want to write some accept, tests.
I'm usually going to actually spend some time designing a text format that describes the internal state of my system in enough detail that I can actually test the important things, but condensed enough so that someone reading over the code can understand it.
And that's one of the first things that I do and then, you know, I tweak it as I go ahead and, you know, see more test cases.
So as you can imagine, accept tests have been a little sort of project of mine regarding testing for a while in PyTorch.
I added a really simple version of them that just wrote things to files at the very beginning People sort of used it.
They didn't like it very much because things were out in another file.
And when I was writing g h stack, see my previous podcast about stacked diff development in Pytorch.
One of the things that I needed to do was write a test suite for g h stack.
And this was kind of not so easy.
Right? Because what is Jade Stack doing? Jade Stack is taking a bunch of commits and then pushing them to GitHub.
And, like, well, for one, like, how do you even test in this situation? Right? You don't want your test suite to be creating tons of repositories and issues on GitHub? By the way, we solve this by, like, creating a crappy fake in memory implementation of GitHub's GraphQL and REST.
like, APIs so that we could mock fake implementations so that the tests could actually run against them.
But that's a story for another day.
And then once you've done all that stuff, you need to actually like stand back and be like, hey, what the heck happened? Did g h stack actually create the pull requests that were necessary? and push the commits that were necessary.
And that was sort of the point where I was like, okay, I'm actually gonna sit down and spend some time writing a module that will help me do inline expect test.
And then I'll also sit down and write a representation for the state of git repositories and pull request so that I can write these tests in a straightforward way.
And so I did.
And the way that the implementation worked was that when a test failed, we would catch the exception that was raised by the error.
And this this back trace would contain a line number to the call to assert expected in line.
That that's the name of the method on the test class that expects test give you.
a back trace with a line number of the thing in question.
And then what we do is we go open up the Python file go to that line and search for the string in question.
And so there's a convention that I picked for our expect test implementation, which is that we only ever substitute triple quoted strings.
In principle, we could substitute single quoted strings, but then it might be easy to end up in a situation where you have like multiple strings on a line.
And so then it's like which one do you replace? triple coated strings don't really have that problem.
You're not very likely to have multiple triple coated strings in one line.
So we find the triple coated string and we do the substitution on it.
and then we write out the Python file back to the end.
And that's basically the crux of how it all works.
There's a funny implementation detail about pre Python three point eight, which is that part of Python three point eight, the actual line number for the back trace is for the end of the statement in question.
So if you have a multi line statement, it gives you a pointer to the last line of the statement in question So you actually have to run the records backwards.
You you're like, okay.
Well, starting from the end, look for the string in question, and then do the such institution.
Details.
But once you do that, all that, you have a implementation of extract test.
you have you do a simple, good old fashioned string comparison to check if the value equals the string in question.
And if you don't like it or you wanna update it, then you just do this reg x on the source file, python source file, and it updates it, and then you can go and take a look in your favorite get diff viewer to see what the change is.
And this is really really easy and makes it super great for, like, you know, writing tests without having to laboriously write down all the things you expect.
So iLoud expects tests a lot They're really powerful and they let you write tests in a lot less time, especially if you write a lot of tests that involve setting up some state running something and then looking at what the results are.
A few words of advice for when you're setting up accept test.
So I've already talked about some of the common problems.
Right? So one is that you don't want the representation to be too long because then it's gonna be like, oh my god.
like, what is all this stuff.
So you want the representation to be actually legible by humans, and that means you have to spend some time designing it.
Some other more basic things that you need to be careful about when you're doing expect us is on one, you need to make sure the output's actually deterministic.
Right? Like, if you're putting a time stamp in the output, That's bad because, well, it's gonna change every time and your expect test is just not going to work.
If you're like writing the output from scratch, this is not a big deal.
Right? You just don't put time stamps in.
But sometimes, there's nondeterministic in your algorithm.
Like, for example, automatic differentiation in Python and PyTorch runs in a multi threaded fashion.
So it's not guaranteed what order your backward nodes will run and is.
So if you're, like, your trace involves like, you know, recording logs when things get run.
Well, just be aware that, you know, that is non deterministic and you might have to do some canonicalization for example, to make sure the expect test all works out.
Sometimes, you can just sort of mask out the text that you don't like.
So it's like, hey, I know this thing is not deterministic.
So before I do the string comparison, I'm gonna go ahead and replace it with some placeholder token and that token will always be the same no matter what I'm doing.
By the way, it's a pretty good idea to make sure your code is deterministic pays off in a lot of other ways.
And so ease of use would expect us is just to get another payoff.
Okay.
So, nuts and bolts of using expect tests in Pytorch.
The default test case that Pytorch provide already contains expect test functionality.
So all you need to do is call the relevant function, and the most common one you'll use is self dot assert expected in line.
Assert expected means it's gonna be an expect test, and in line means that you're gonna put the string directly in line inside of your source code program.
There's also variants that work for if an exception is raised What do you expect the expect exception text to be? Just check the expect test module in Pytorch to see what API options are available to you.
The module that implements expect test itself is actually pretty self contained and I copy pasted it between GH Stack and PyTorch because I didn't feel like making a separate package to do this.
But if this is a code you are interested in, shoot me a tweet and I'll figure out what I can do about actually publishing it for real.
That's everything I wanted to say for today.
Talk to you next time.
.
