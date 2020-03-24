





SELECT listing_id, 
	ROUND(AVG(pol_pos_score),4) as avg_pol_pos_score, 
	ROUND(AVG(pol_neg_score),4) as avg_pol_neg_score, 
	ROUND(AVG(pol_neu_score),4) as avg_pol_neu_score, 
	ROUND(AVG(pol_compound_score),4) as avg_pol_compound_score,
	case 
		when ROUND(AVG(pol_compound_score),4) >= 0.05 then 'positive' 
		when ROUND(AVG(pol_compound_score),4) > -0.05 and ROUND(AVG(pol_compound_score),4) < 0.05 then 'neutral'
		else 'negative' end as avg_review_sentiment
INTO [mar653_airbnb_seattle].[dev_brook].ListingReviewsAvgSentimentScores
FROM [mar653_airbnb_seattle].[dev_brook].[ReviewsSentimentScores]
GROUP BY listing_id
ORDER BY avg_pol_compound_score desc
;


select listing_id
	,COUNT(*) as available_days
	,AVG(price) as avg_price
from [mar653_airbnb_seattle].[dev_brook].[CalendarListings]
where available <> 0
group by listing_id
order by available_days desc
;