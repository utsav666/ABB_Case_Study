import matplotlib
from pandas import DataFrame

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.agent.chart_gen_agent import ChartAgent
from app.agent.fall_back import GeneralAnswerAgent
from app.agent.schema_info_agent import SchemaInfoAgent
from app.agent.sql_generation_agent import SQLGenerationAgent
from app.agent.table_identification_agent import TableIdentiferAgent
from app.router.route_improvisation import Router
from app.utils.char_context import ChartContext
from app.utils.sql_executor import SqlExec
from app.utils.sql_validator import validator


######Simple Hardcoded Orchetrator########
class Orchestrator:
    def __init__(self, user_question):
        self.user_question = user_question
        self.TA = TableIdentiferAgent(self.user_question)
        self.router = Router()

    def executor(self):
        print("user question .... " + self.user_question)
        print("\n.....Intent identifcation......")
        decision_maker = self.router.route_question(self.user_question)
        # print(memory,"........mory........")
        print(decision_maker, "decision_maker")
        if decision_maker.route == "sql_query":
            ta_agent_result = self.TA.run()
            print("\n....Table identifier result... " + str(ta_agent_result))
            print("\nRunning SQL Generation Agent")
            sql_gen_agent = SQLGenerationAgent(
                self.user_question, ta_agent_result["relevant_tables"]
            )
            sql_gen_res = sql_gen_agent.run()
            print("\n....SQL generation Result... " + sql_gen_res)
            sql_val = validator(sql_gen_res)
            if sql_val is True:
                print("\n.........executing query......")
                sql_exec = SqlExec().sql_executor(sql_gen_res)
                # print(sql_exec.shape)
                return sql_exec
            else:
                print("\nFailed to generate sql")
        elif decision_maker.route == "sql_and_chart":
            ta_agent_result = self.TA.run()
            print("\n....Table identifier result... " + str(ta_agent_result))
            print("\nRunning SQL Generation Agent")
            sql_gen_agent = SQLGenerationAgent(
                self.user_question, ta_agent_result["relevant_tables"]
            )
            sql_gen_res = sql_gen_agent.run()
            print("\n....SQL generation Result... " + sql_gen_res)
            sql_val = validator(sql_gen_res)
            if sql_val is True:
                print("\n.........executing query......")
                sql_exec: DataFrame | str = SqlExec().sql_executor(sql_gen_res)
                if not isinstance(sql_exec, DataFrame):
                    print("SQL execution did not return a DataFrame:", repr(sql_exec))
                    return {
                        "type": "error",
                        "message": "Query execution failed or returned invalid result.",
                    }

                chart_context = ChartContext.build_chart_context(sql_exec)
                chart_gen_decision = ChartAgent().run(self.user_question, chart_context)
                print(chart_gen_decision, "....chart gen decision....")

                if (
                    chart_gen_decision.x_axis not in sql_exec.columns
                    or chart_gen_decision.y_axis not in sql_exec.columns
                ):
                    print(
                        "Chart axis not found in SQL result columns:",
                        sql_exec.columns.tolist(),
                    )
                    return {
                        "type": "error",
                        "message": "Chart axis names not present in query result.",
                    }

                fig, ax = plt.subplots()
                if chart_gen_decision.chart_type == "line":
                    ax.plot(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],
                    )
                elif chart_gen_decision.chart_type == "bar":
                    ax.bar(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],
                    )
                elif chart_gen_decision.chart_type == "scatter":
                    ax.scatter(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],
                    )
                elif chart_gen_decision.chart_type == "histogram":
                    ax.hist(sql_exec[chart_gen_decision.x_axis], bins=10)
                elif chart_gen_decision.chart_type == "pie":
                    ax.pie(
                        sql_exec[chart_gen_decision.y_axis],
                        labels=sql_exec[chart_gen_decision.x_axis].astype(str).tolist(),
                        autopct="%1.1f%%",
                    )
                elif chart_gen_decision.chart_type == "box":
                    ax.boxplot(sql_exec[chart_gen_decision.y_axis])
                elif chart_gen_decision.chart_type == "area":
                    ax.fill_between(
                        sql_exec[chart_gen_decision.x_axis],
                        sql_exec[chart_gen_decision.y_axis],  # type: ignore
                    )
                ax.set_xlabel(chart_gen_decision.x_axis)
                ax.set_ylabel(chart_gen_decision.y_axis)
                return {"type": "chart", "figure": fig, "data": sql_exec}
            else:
                print("\nFailed to generate charts")

        elif decision_maker.route == "configuration":
            return (
                "Use the `/schema/config` FastAPI endpoints to manage column "
                "labels, descriptions, and assistant visibility."
            )
        elif decision_maker.route == "schema_info":
            schema_info_agent = SchemaInfoAgent(
                self.user_question, decision_maker.reasoning
            )
            res = schema_info_agent.run()
            print(res)
            return res
        else:
            gen_ans_agent = GeneralAnswerAgent()
            res = gen_ans_agent.run("...", self.user_question, decision_maker.reasoning)
            return res


#### Unite Test 3 #########
# if __name__ == "__main__":
#     user_query = "visuzlie me year wise movie count"
#     print("..Orchestration Started.....")
#     orch_run = Orchestrator(user_query)
#     orch_run.executor()
#     print("..Orchetsration Completed.....")
