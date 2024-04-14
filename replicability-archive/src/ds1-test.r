df <- read.csv("data/ds1-you-thank-again.csv")

t <- table(df$you_as_subject, df$c2_thankfulness)

print(t)

print(chisq.test(t,correct=FALSE))

print(t[2,2] / (t[2,2] + t[2,1]))
print(t[1,2] / (t[1,1] + t[1,2]))

no = df[df$you_as_subject == 0, ]
ye = df[df$you_as_subject == 1, ]

print(paste("num no you_as_subject", length(no$you_as_subject)))
print(paste("num ye you_as_subject", length(ye$you_as_subject)))
print(paste("num total", length(df$you_as_subject)))

print(paste("num no you_as_subject no c2_thankfulness", length(no[no$c2_thankfulness == 0, ]$you_as_subject)))
print(paste("num no you_as_subject ye c2_thankfulness", length(no[no$c2_thankfulness == 1, ]$you_as_subject)))
print(paste("num total", length(df$you_as_subject)))

print(mean(ye$c2_thankfulness))
print(mean(no$c2_thankfulness))
