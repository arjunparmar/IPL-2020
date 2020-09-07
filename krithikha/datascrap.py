import csv
import pandas as pd
deliveries=pd.read_csv("deliveries.csv")
print(deliveries.shape[0])
with open('deliveries.csv','r') as csv_file:
	csv_reader=csv.reader(csv_file)
	reader=csv.DictReader(csv_file)

	with open('new2.csv','w',newline="") as new:
		field=['batting_team','bowling_team', 'noball_runs', 'penalty_runs', 'dismissal_kind', 'match_id','is_super_over','over', 'non_striker', 'player_dismissed', 'wide_runs', 'inning', 'extra_runs', 'legbye_runs', 'ball', 'batsman_runs', 'bowler', 'bye_runs', 'batsman', 'fielder','total_runs']
		csv_writer=csv.DictWriter(new,fieldnames=field,delimiter=',')
		csv2=csv.writer(new,delimiter=",")
		#csv_writer.writeheader()
		s=0
		#for i in range (1,deliveries.shape[0]):

		for row in reader:
				del row['ball'],row['noball_runs'],row['penalty_runs'],row['match_id'],row['batsman_runs'],row['batsman'],row['legbye_runs'],row['dismissal_kind'],row['non_striker'],row['bowler'],row['player_dismissed'],row['wide_runs'],row['bye_runs'],row['fielder']
				if(int(row['inning'])==2):
					#print(row['over'])

					if(int(row['is_super_over'])==0):
						if(int(row['over']) in range (0,7)):
							s=s+int(row['total_runs'])

							csv_writer.writerow(row)
							deliveries["powerplay_runs"]=s
							rown=next(csv_reader)
							print("reached","sdfffffffwwwwwwwwwwwwww",rown['over'])
							if(int(row['over'])==6) and (int(rown['over'])==1):
								print("jewije",s)
								s=0

				#break 

	

csv_file.close()