
setwd("C:/Users/Bill/Desktop/Tahzoo.com Redesign/GoogleAnalytics/Experiments")
library(RGoogleAnalytics)

load("C:/Users/Bill/Documents/google-analytics/GoogleAnalytics/toekn_file")
ValidateToken(token)

days_back <- 120
ql = Init(start.date = format(Sys.Date()-days_back,"%Y-%m-%d"), #note the super clever way of counting days
            end.date = format(Sys.Date(),"%Y-%m-%d"),  #you can enter specific dates as well. 
            metrics =  "ga:sessions,ga:totalEvents",
            dimensions = "ga:eventLabel",
            max.results = 10000,            #10000 is the max, you will have to paginate your query
            table.id = "ga:56928074")

q2 = Init(start.date = format(Sys.Date()-days_back,"%Y-%m-%d"), 
            end.date = format(Sys.Date(),"%Y-%m-%d"),
            metrics =  "ga:users",
            max.results = 10000,   
            table.id = "ga:56928074")

c("from: ", format(Sys.Date()-days_back,"%Y-%m-%d"), " To: ", format(Sys.Date(),"%Y-%m-%d"))



gq = QueryBuilder(ql)
gq2 = QueryBuilder(q2)

gd = GetReportData(gq, token, paginate_query = F)
users = GetReportData(gq2, token)

head(gd)

c("Total sessions with no GA code", gd[gd["eventLabel"]=="","sessions"]/sum(gd$sessions))
c("Total sessions in dataset",sum(gd$sessions))

c("Number of unique visitors who are tracked",length(unique(gd$eventLabel))-1)

gd$eventsPerSession <- gd$totalEvents/gd$sessions
summary(gd)

clean_gd <- gd[gd$eventLabel!="",]
summary(clean_gd)

users

c("Percent of total users who do not have a tracking code: ",
      users[1,"users"]-length(unique(clean_gd$eventLabel)),
      1-(length(unique(clean_gd$eventLabel))/users[1,"users"]))

q3 = Init(start.date = format(Sys.Date()-days_back,"%Y-%m-%d"), 
            end.date = format(Sys.Date(),"%Y-%m-%d"),
            metrics =  "ga:users",   #users is just a placeholder here, I just want a list of events and labels
            dimensions = "ga:eventLabel, ga:eventCategory",
            max.results = 10000,   
            filters = "ga:eventCategory!=::",  #filtering out events where the lable is blank.  "::" is a blank Category 
            table.id = "ga:56928074")

gq3 = QueryBuilder(q3)
events = GetReportData(gq3, token, paginate_query = T)


summary(events)

head(events)

write.table(events, file = "labels_categories_actions.tsv", sep = "\t",
            fileEncoding = "utf-8")


